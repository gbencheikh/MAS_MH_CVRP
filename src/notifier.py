
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def success(message):
    """Message de succès"""
    print("\n" + "="*60)
    print(f"{Colors.OKGREEN}[{timestamp()}] 🎉 {Colors.ENDC} {message}")
    print("="*60 + "\n")
    
def notify(message):
    print(f"{Colors.OKCYAN} [{timestamp()}] 💬 {message}{Colors.ENDC}")

def passed(message):
    print(f"\n {Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def warning(message):
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")

def error(message):
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def saved(message, filepath):
    print(f"{Colors.OKBLUE}[{timestamp()}] 💾 {message} : {filepath}{Colors.ENDC}")

def configuration(): 
    print("⚙️  Configuration...")
    
def section(message): 
    print("=" * 60)
    print(message)
    print("=" * 60)
    print()