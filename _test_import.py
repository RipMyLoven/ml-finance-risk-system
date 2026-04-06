import sys
sys.path.insert(0, r'C:\Users\ripmy\Documents\GitHub\ml-finance-risk-system')
try:
    from tui.config import AppConfig
    print('OK: config imported')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
