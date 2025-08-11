class VectorDBError(Exception):
    def __init__(self, db_name: str, message: str):
        self.db_name = db_name
        self.message = message
        super().__init__(f"Error with {self.db_name}: {self.message}")
