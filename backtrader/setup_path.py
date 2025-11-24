"""
設定 Python 路徑，讓 notebook 可以引用專案根目錄的 alanq 模組
在所有 notebook 的第一個 cell 執行: import setup_path
"""
import sys
from pathlib import Path

# 獲取專案根目錄（notebooks 的父目錄）
# 這個檔案位於 notebooks/setup_path.py，所以 parent 就是專案根目錄
project_root = Path(__file__).parent.parent

# 將專案根目錄加入 Python 路徑
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 可選：驗證設定（開發時可用，正式使用時可註解掉）
# print(f"✓ 已將專案根目錄加入路徑: {project_root}")
# print(f"✓ alanq 模組位置: {project_root / 'alanq'}")


