# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import psycopg
from langgraph.store.memory import InMemoryStore
from psycopg.rows import dict_row
from pymongo import MongoClient

from src.config.loader import get_bool_env, get_str_env


class ChatStreamManager:
    """
    管理聊天流消息，提供持久化存储和内存缓存功能。

    此类使用内存存储临时数据和MongoDB或PostgreSQL进行持久化存储，
    处理聊天消息的存储和检索。它跟踪消息块，并在对话结束时整合它们。

    Attributes:
        store (InMemoryStore): 用于临时消息块的内存存储
        mongo_client (MongoClient): MongoDB客户端连接
        mongo_db (Database): MongoDB数据库实例
        postgres_conn (psycopg.Connection): PostgreSQL连接
        logger (logging.Logger): 此类的日志记录器实例
    """

    def __init__(
        self, checkpoint_saver: bool = False, db_uri: Optional[str] = None
    ) -> None:
        """
        使用数据库连接初始化ChatStreamManager。

        Args:
            checkpoint_saver: 是否启用检查点保存功能
            db_uri: 数据库连接URI。支持MongoDB (mongodb://) 和PostgreSQL (postgresql://)
                   如果为None，则使用LANGGRAPH_CHECKPOINT_DB_URL环境变量或默认为localhost
        """
        self.logger = logging.getLogger(__name__)
        self.store = InMemoryStore()
        self.checkpoint_saver = checkpoint_saver
        # 使用提供的URI或回退到环境变量或默认值
        self.db_uri = db_uri

        # 初始化数据库连接
        self.mongo_client = None
        self.mongo_db = None
        self.postgres_conn = None

        if self.checkpoint_saver:
            if self.db_uri.startswith("mongodb://"):
                self._init_mongodb()
            elif self.db_uri.startswith("postgresql://") or self.db_uri.startswith(
                "postgres://"
            ):
                self._init_postgresql()
            else:
                self.logger.warning(
                    f"不支持的数据库URI方案: {self.db_uri}. "
                    "支持的方案: mongodb://, postgresql://, postgres://"
                )
        else:
            self.logger.warning("检查点保存器已禁用")

    def _init_mongodb(self) -> None:
        """初始化MongoDB连接。"""

        try:
            self.mongo_client = MongoClient(self.db_uri)
            self.mongo_db = self.mongo_client.checkpointing_db
            # 测试连接
            self.mongo_client.admin.command("ping")
            self.logger.info("成功连接到MongoDB")
        except Exception as e:
            self.logger.error(f"连接MongoDB失败: {e}")

    def _init_postgresql(self) -> None:
        """初始化PostgreSQL连接并在需要时创建表。"""

        try:
            self.postgres_conn = psycopg.connect(self.db_uri, row_factory=dict_row)
            self.logger.info("成功连接到PostgreSQL")
            self._create_chat_streams_table()
        except Exception as e:
            self.logger.error(f"连接PostgreSQL失败: {e}")

    def _create_chat_streams_table(self) -> None:
        """如果chat_streams表不存在则创建它。"""
        try:
            with self.postgres_conn.cursor() as cursor:
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS chat_streams (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    thread_id VARCHAR(255) NOT NULL UNIQUE,
                    messages JSONB NOT NULL,
                    ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_chat_streams_thread_id ON chat_streams(thread_id);
                CREATE INDEX IF NOT EXISTS idx_chat_streams_ts ON chat_streams(ts);
                """
                cursor.execute(create_table_sql)
                self.postgres_conn.commit()
                self.logger.info("聊天流表创建/验证成功")
        except Exception as e:
            self.logger.error(f"创建聊天流表失败: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()

    def process_stream_message(
        self, thread_id: str, message: str, finish_reason: str
    ) -> bool:
        """
        处理并存储聊天流消息块。

        此方法处理流式传输期间的各个消息块，并在流完成时将它们整合为完整的消息。
消息临时存储在内存中，完成后永久保存在MongoDB中。

        Args:
            thread_id: 对话线程的唯一标识符
            message: 要存储的消息内容或块
            finish_reason: 消息完成的原因 ("stop", "interrupt", 或部分完成)

        Returns:
            bool: 如果消息处理成功返回True，否则返回False
        """
        if not thread_id or not isinstance(thread_id, str):
            self.logger.warning("提供了无效的thread_id")
            return False

        if not message:
            self.logger.warning("提供了空消息")
            return False

        try:
            # 为此线程的消息创建命名空间
            store_namespace: Tuple[str, str] = ("messages", thread_id)

            # 获取或初始化用于跟踪块的消息游标
            cursor = self.store.get(store_namespace, "cursor")
            current_index = 0

            if cursor is None:
                # 为新对话初始化游标
                self.store.put(store_namespace, "cursor", {"index": 0})
            else:
                # 为下一个块增加索引
                current_index = int(cursor.value.get("index", 0)) + 1
                self.store.put(store_namespace, "cursor", {"index": current_index})

            # 存储当前消息块
            self.store.put(store_namespace, f"chunk_{current_index}", message)

            # 检查对话是否完成并应持久化
            if finish_reason in ("stop", "interrupt"):
                return self._persist_complete_conversation(
                    thread_id, store_namespace, current_index
                )

            return True

        except Exception as e:
            self.logger.error(
                f"处理线程 {thread_id} 的流消息时出错: {e}"
            )
            return False

    def _persist_complete_conversation(
        self, thread_id: str, store_namespace: Tuple[str, str], final_index: int
    ) -> bool:
        """
        将完成的对话持久化到数据库（MongoDB或PostgreSQL）。

        从内存存储中检索所有消息块，并将完整的对话保存到配置的数据库中进行永久存储。

        Args:
            thread_id: 对话线程的唯一标识符
            store_namespace: 用于访问存储消息的命名空间元组
            final_index: 此对话的最终块索引

        Returns:
            bool: 如果持久化成功返回True，否则返回False
        """
        try:
            # 从内存存储中检索所有消息块
            # 获取直到最终索引的所有消息，包括游标元数据
            memories = self.store.search(store_namespace, limit=final_index + 2)

            # 提取消息内容，过滤掉游标元数据
            messages: List[str] = []
            for item in memories:
                value = item.dict().get("value", "")
                # 跳过游标元数据，只包括实际的消息块
                if value and not isinstance(value, dict):
                    messages.append(str(value))

            if not messages:
                self.logger.warning(f"未找到线程 {thread_id} 的消息")
                return False

            if not self.checkpoint_saver:
                self.logger.warning("检查点保存器已禁用")
                return False

            # 根据可用连接选择持久化方法
            if self.mongo_db is not None:
                return self._persist_to_mongodb(thread_id, messages)
            elif self.postgres_conn is not None:
                return self._persist_to_postgresql(thread_id, messages)
            else:
                self.logger.warning("没有可用的数据库连接")
                return False

        except Exception as e:
            self.logger.error(
                f"持久化线程 {thread_id} 的对话时出错: {e}"
            )
            return False

    def _persist_to_mongodb(self, thread_id: str, messages: List[str]) -> bool:
        """将对话持久化到MongoDB。"""
        try:
            # 获取聊天流的MongoDB集合
            collection = self.mongo_db.chat_streams

            # 检查数据库中是否已存在对话
            existing_document = collection.find_one({"thread_id": thread_id})

            current_timestamp = datetime.now()

            if existing_document:
                # 使用新消息更新现有对话
                update_result = collection.update_one(
                    {"thread_id": thread_id},
                    {"$set": {"messages": messages, "ts": current_timestamp}},
                )
                self.logger.info(
                    f"更新线程 {thread_id} 的对话: "
                    f"{update_result.modified_count} 个文档被修改"
                )
                return update_result.modified_count > 0
            else:
                # 创建新的对话文档
                new_document = {
                    "thread_id": thread_id,
                    "messages": messages,
                    "ts": current_timestamp,
                    "id": uuid.uuid4().hex,
                }
                insert_result = collection.insert_one(new_document)
                self.logger.info(
                    f"创建新对话: {insert_result.inserted_id}"
                )
                return insert_result.inserted_id is not None

        except Exception as e:
            self.logger.error(f"持久化到MongoDB时出错: {e}")
            return False

    def _persist_to_postgresql(self, thread_id: str, messages: List[str]) -> bool:
        """将对话持久化到PostgreSQL。"""
        try:
            with self.postgres_conn.cursor() as cursor:
                # 检查是否已存在对话
                cursor.execute(
                    "SELECT id FROM chat_streams WHERE thread_id = %s", (thread_id,)
                )
                existing_record = cursor.fetchone()

                current_timestamp = datetime.now()
                messages_json = json.dumps(messages)

                if existing_record:
                    # 使用新消息更新现有对话
                    cursor.execute(
                        """
                        UPDATE chat_streams 
                        SET messages = %s, ts = %s 
                        WHERE thread_id = %s
                        """,
                        (messages_json, current_timestamp, thread_id),
                    )
                    affected_rows = cursor.rowcount
                    self.postgres_conn.commit()

                    self.logger.info(
                        f"更新线程 {thread_id} 的对话: "
                        f"{affected_rows} 行被修改"
                    )
                    return affected_rows > 0
                else:
                    # 创建新的对话记录
                    conversation_id = uuid.uuid4()
                    cursor.execute(
                        """
                        INSERT INTO chat_streams (id, thread_id, messages, ts) 
                        VALUES (%s, %s, %s, %s)
                        """,
                        (conversation_id, thread_id, messages_json, current_timestamp),
                    )
                    affected_rows = cursor.rowcount
                    self.postgres_conn.commit()

                    self.logger.info(
                        f"创建ID为 {conversation_id} 的新对话"
                    )
                    return affected_rows > 0

        except Exception as e:
            self.logger.error(f"持久化到PostgreSQL时出错: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
            return False

    def close(self) -> None:
        """关闭数据库连接。"""
        try:
            if self.mongo_client is not None:
                self.mongo_client.close()
                self.logger.info("MongoDB连接已关闭")
        except Exception as e:
            self.logger.error(f"关闭MongoDB连接时出错: {e}")

        try:
            if self.postgres_conn is not None:
                self.postgres_conn.close()
                self.logger.info("PostgreSQL连接已关闭")
        except Exception as e:
            self.logger.error(f"关闭PostgreSQL连接时出错: {e}")

    def __enter__(self):
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 - 关闭连接。"""
        self.close()


# 向后兼容的全局实例
# TODO: 考虑使用依赖注入代替全局实例
_default_manager = ChatStreamManager(
    checkpoint_saver=get_bool_env("LANGGRAPH_CHECKPOINT_SAVER", False),
    db_uri=get_str_env("LANGGRAPH_CHECKPOINT_DB_URL", "mongodb://localhost:27017"),
)


def chat_stream_message(thread_id: str, message: str, finish_reason: str) -> bool:
    """
    向后兼容的遗留函数包装器。

    Args:
        thread_id: 对话线程的唯一标识符
        message: 要存储的消息内容
        finish_reason: 消息完成的原因

    Returns:
        bool: 如果消息处理成功返回True
    """
    checkpoint_saver = get_bool_env("LANGGRAPH_CHECKPOINT_SAVER", False)
    if checkpoint_saver:
        return _default_manager.process_stream_message(
            thread_id, message, finish_reason
        )
    else:
        return False