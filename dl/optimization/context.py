

class RunContext:
    def __init__(self):
        self.run = True
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            self.run = False
            
            return True
        
        elif exc_type:
            return False
        
        else:
            return True
