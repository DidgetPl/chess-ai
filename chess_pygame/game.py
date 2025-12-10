import typing

import chess
from evaluation_model import EvaluationModel
from model_loader import load_model
from negamax_module import choose_move_negamax


class ChessGame:
    def __init__(self, model_name, device="cpu", ai_depth=2):
        self.board = chess.Board()
        self.device = device
        self.ai_depth = ai_depth
        self.model_name = model_name
        self.model = load_model(model_name, device)
        self.move_counter = 0
        self.last_move_player = None
        self.last_move_ai = None
        self.game_over_msg = None

    def restart(self):
        self.board.reset()
        self.move_counter = 0
        self.last_move_player = None
        self.last_move_ai = None
        self.game_over_msg = None

    def set_model(self, model_name):
        self.model_name = model_name
        self.model = load_model(model_name, self.device)

    def player_move(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
            self.last_move_player = move
            self.move_counter += 1
            self.game_over_msg = self.check_game_over()
            return True
        return False

    def ai_move(self):
        if self.game_over_msg:
            return
        move, score = choose_move_negamax(self.model, self.board, depth=self.ai_depth, device=self.device)
        if move:
            self.board.push(move)
            self.last_move_ai = move
            self.move_counter += 1
        self.game_over_msg = self.check_game_over()

    def check_game_over(self):
        b = self.board
        if b.is_checkmate():
            return f"Szach-mat! {'Biały' if b.turn == chess.BLACK else 'Czarny'} wygrywa"
        if b.is_stalemate():
            return "Pat — remis"
        if b.is_insufficient_material():
            return "Remis — za mało materiału"
        if b.can_claim_threefold_repetition():
            return "Remis — trzykrotne powtórzenie"
        if b.can_claim_fifty_moves():
            return "Remis — zasada 50 ruchów"
        return None
