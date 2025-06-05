import keyboard
import time
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class KeyboardController:
    def __init__(self):
        self.registered_shortcuts = {}
        
    def type_text(self, text: str) -> str:
        """Type text using keyboard"""
        try:
            keyboard.write(text)
            return f"Typed text: {text}"
        except Exception as e:
            return f"Error typing text: {str(e)}"
            
    def press_key(self, key: str) -> str:
        """Press a single key"""
        try:
            keyboard.press_and_release(key)
            return f"Pressed key: {key}"
        except Exception as e:
            return f"Error pressing key: {str(e)}"
            
    def hold_key(self, key: str) -> str:
        """Hold down a key"""
        try:
            keyboard.press(key)
            return f"Holding key: {key}"
        except Exception as e:
            return f"Error holding key: {str(e)}"
            
    def release_key(self, key: str) -> str:
        """Release a held key"""
        try:
            keyboard.release(key)
            return f"Released key: {key}"
        except Exception as e:
            return f"Error releasing key: {str(e)}"
            
    def press_hotkey(self, *keys: str) -> str:
        """Press a combination of keys"""
        try:
            keyboard.press_and_release('+'.join(keys))
            return f"Pressed hotkey: {'+'.join(keys)}"
        except Exception as e:
            return f"Error pressing hotkey: {str(e)}"
    
    def register_shortcut(self, keys: str, callback) -> str:
        """Register a keyboard shortcut"""
        try:
            keyboard.add_hotkey(keys, callback)
            self.registered_shortcuts[keys] = callback
            return f"Registered shortcut: {keys}"
        except Exception as e:
            return f"Error registering shortcut: {str(e)}"
            
    def unregister_shortcut(self, keys: str) -> str:
        """Unregister a keyboard shortcut"""
        try:
            keyboard.remove_hotkey(keys)
            if keys in self.registered_shortcuts:
                del self.registered_shortcuts[keys]
            return f"Unregistered shortcut: {keys}"
        except Exception as e:
            return f"Error unregistering shortcut: {str(e)}"
    
    def get_active_shortcuts(self) -> List[str]:
        """Get list of active shortcuts"""
        return list(self.registered_shortcuts.keys())
    
    # Common keyboard shortcuts
    def copy(self) -> str:
        """Copy selected text"""
        return self.press_hotkey('ctrl', 'c')
        
    def paste(self) -> str:
        """Paste copied text"""
        return self.press_hotkey('ctrl', 'v')
        
    def cut(self) -> str:
        """Cut selected text"""
        return self.press_hotkey('ctrl', 'x')
        
    def select_all(self) -> str:
        """Select all text"""
        return self.press_hotkey('ctrl', 'a')
        
    def undo(self) -> str:
        """Undo last action"""
        return self.press_hotkey('ctrl', 'z')
        
    def redo(self) -> str:
        """Redo last undone action"""
        return self.press_hotkey('ctrl', 'y')
        
    def save(self) -> str:
        """Save current file"""
        return self.press_hotkey('ctrl', 's')
        
    def find(self) -> str:
        """Open find dialog"""
        return self.press_hotkey('ctrl', 'f')
        
    def switch_window(self) -> str:
        """Switch between windows"""
        return self.press_hotkey('alt', 'tab')
        
    def close_window(self) -> str:
        """Close current window"""
        return self.press_hotkey('alt', 'f4')

    def cleanup(self):
        """Clean up registered shortcuts"""
        for keys in list(self.registered_shortcuts.keys()):
            self.unregister_shortcut(keys)

# Create global keyboard controller instance
keyboard_controller = KeyboardController()
