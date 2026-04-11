import pymem
import pymem.process
import time



def _init_memory_reader():
    """初始化 pymem 並設定指針路徑"""
    try:
        global pm, game_module, static_offset, offsets
        process_name = "SAWBLADES_DEMO.exe"
        pm = pymem.Pymem(process_name)
        print(f"已連接到遊戲進程: {process_name}")

        # 取得模組基址
        game_module = pymem.process.module_from_name(pm.process_handle, process_name).lpBaseOfDll
        
        # Base Offset
        static_offset = 0x00ED7200
        offsets = [0x20, 0x18, 0x0, 0x8, 0x10, 0x390, 0x0]
        
    except Exception as e:
        print(f"記憶體讀取初始化失敗: {e}")
        pm = None
            
def get_score_from_memory():
    """從記憶體讀取當前分數"""
    if pm is None:
        print("Pymem 未初始化")
        return 0
    try:
        # 1. 讀取基底位址
        addr = game_module + static_offset
        # 2. 逐層讀取指針
        for offset in offsets:
            addr = pm.read_longlong(addr)
            if addr == 0:
                print("指針為空，無法讀取分數")
                return 0
            addr = addr + offset
        score = pm.read_double(addr)
        return score
        
    except Exception as e:
        # print(f"讀取分數失敗: {e}")
        return 0
        
if __name__ == "__main__":
    _init_memory_reader()
    time.sleep(2)
    while True:
        score = get_score_from_memory()
        print(f"當前分數: {score}")