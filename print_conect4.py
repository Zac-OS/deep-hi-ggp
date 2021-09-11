class PrintConect4:
    def __init__(self, game):
        size = 0
        if game == "blindtictactoe":
            size = 3
        if game == "kriegTTT_5x5":
            size = 5
        self.size = size
        self.board = [[' ' for i in range(size)] for j in range(size)]

    def reset(self):
        self.board = [[' ' for i in range(self.size)] for j in range(self.size)]

    def make_moves(self, moves, propnet):
        if self.size == 0:
            return
        printing_moves = {}
        for role in propnet.roles:
            for move in propnet.legal_for[role]:
                if move.id in moves:
                    printing_moves[role] = move.gdl
        # print(printing_moves)
        if "oplayer" in printing_moves:
            printing_moves["o"] = printing_moves["oplayer"].replace("oplayer", "o")
            printing_moves["x"] = printing_moves["xplayer"].replace("xplayer", "x")
        assert "o" in printing_moves
        x0, x1 = int(printing_moves["x"][17])-1, int(printing_moves["x"][19])-1
        o0, o1 = int(printing_moves["o"][17])-1, int(printing_moves["o"][19])-1
        if (x0, x1) == (o0, o1):
            if "random" in printing_moves:
                if "x" in printing_moves["random"]:
                    self.board[x0][x1] = "x"
                else:
                    self.board[o0][o1] = "o"
        else:
            if self.board[x0][x1] == " ":
                self.board[x0][x1] = "x"
            if self.board[o0][o1] == " ":
                self.board[o0][o1] = "o"

    def print_moves(self):
        for line in self.board:
            print("|".join(line))
