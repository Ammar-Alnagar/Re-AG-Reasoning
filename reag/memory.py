from pydantic import BaseModel
from typing import List, Dict, Union


class Memory(BaseModel):
    history: List[Dict[str, Union[str, int]]] = []

    def add_message(self, message: Dict[str, Union[str, int]]):
        self.history.append(message)

    def get_history(self) -> List[Dict[str, Union[str, int]]]:
        return self.history

    def clear(self):
        self.history = []
