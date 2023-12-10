import ast
import logging

import torch
from main import inference

logger = logging.getLogger(__name__)

OP_MAP = {
    ast.Add: torch.add,
    ast.Sub: torch.sub,
}


class Calc(ast.NodeVisitor):
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        try:
            return OP_MAP[type(node.op)](left, right)
        except KeyError:
            raise NotImplementedError(f"Operator {type(node.op)} not implemented")

    def embed(self, word):
        with torch.no_grad():
            return inference.embed(word)

    def visit_Str(self, node):
        return self.embed(node.s)

    def visit_Name(self, node):
        return self.embed(node.id)

    def visit_Expr(self, node):
        return self.visit(node.value)

    @classmethod
    def evaluate(cls, expression):
        tree = ast.parse(expression)
        calc = cls()
        return calc.visit(tree.body[0])


if __name__ == "__main__":
    logger.info("Enter words and operators to calculate")
    vector = Calc.evaluate(input())
    results = inference.search(vector)
    for word, score in results:
        print(f"{score:.3f}: {word}")
