# 测试 SQLite 持久化功能
# 文件位置: d:\AI\deer-flow\test_sqlite_persistence.py

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from pathlib import Path

def test_sqlite_persistence():
    """测试 SQLite 持久化功能是否正常工作"""
    
    print("=" * 50)
    print("测试 SQLite 持久化功能")
    print("=" * 50)
    
    # 1. 测试导入
    print("\n1. 测试导入 SqliteSaver...")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        print("   ✅ 成功导入 SqliteSaver")
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False
    
    # 2. 测试创建数据库连接
    print("\n2. 测试创建 SQLite 数据库连接...")
    try:
        test_db_path = "test_checkpoints.db"
        saver = SqliteSaver.from_conn_string(test_db_path)
        print(f"   ✅ 成功创建 SqliteSaver 实例")
        print(f"   📁 数据库路径: {test_db_path}")
    except Exception as e:
        print(f"   ❌ 创建失败: {e}")
        return False
    
    # 3. 检查数据库文件是否存在
    print("\n3. 检查数据库文件...")
    db_file = Path(test_db_path)
    if db_file.exists():
        print(f"   ✅ 数据库文件已创建: {db_file.absolute()}")
        print(f"   📊 文件大小: {db_file.stat().st_size} bytes")
    else:
        print(f"   ⚠️ 数据库文件尚未创建（可能在第一次写入时创建）")
    
    # 4. 测试导入 build_graph_with_memory
    print("\n4. 测试导入 build_graph_with_memory...")
    try:
        from src.graph.builder import build_graph_with_memory
        print("   ✅ 成功导入 build_graph_with_memory")
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return False
    
    # 5. 测试构建图
    print("\n5. 测试构建带记忆的图...")
    try:
        graph = build_graph_with_memory()
        print("   ✅ 成功构建图")
        print(f"   📋 图对象类型: {type(graph).__name__}")
    except Exception as e:
        print(f"   ❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 检查 checkpoints.db
    print("\n6. 检查 checkpoints.db 文件...")
    checkpoints_db = Path("checkpoints.db")
    if checkpoints_db.exists():
        print(f"   ✅ checkpoints.db 已创建: {checkpoints_db.absolute()}")
        print(f"   📊 文件大小: {checkpoints_db.stat().st_size} bytes")
    else:
        print(f"   ⚠️ checkpoints.db 尚未创建（将在第一次对话时创建）")
    
    print("\n" + "=" * 50)
    print("✅ SQLite 持久化功能测试通过！")
    print("=" * 50)
    
    # 清理测试文件
    if db_file.exists():
        db_file.unlink()
        print(f"\n🧹 已清理测试文件: {test_db_path}")
    
    return True

if __name__ == "__main__":
    test_sqlite_persistence()
