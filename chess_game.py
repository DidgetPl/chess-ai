import time
import typing

import chess
import chess.engine
import numpy as np
import torch

from evaluation_model import *

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_binary(board: chess.Board) -> torch.Tensor:
    bits = []
    for color in [chess.WHITE, chess.BLACK]:
        for piece in PIECE_TYPES:
            squares = board.pieces(piece, color)
            for sq in chess.SQUARES:
                bits.append(1 if sq in squares else 0)
    bits.append(1 if board.turn == chess.WHITE else 0)
    bits.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    bits.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    bits.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    bits.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    if board.ep_square is None:
        ep_file = 0
    else:
        ep_file = chess.square_file(board.ep_square)
    for i in range(4):
        bits.append((ep_file >> i) & 1)
    for i in range(8):
        bits.append((board.halfmove_clock >> i) & 1)
    while len(bits) < 808:
        bits.append(0)
    arr = np.array(bits, dtype=np.float32)
    return torch.from_numpy(arr)

def eval_position(model: torch.nn.Module, board: chess.Board, device='cpu') -> float:
    model.eval()
    x = board_to_binary(board).to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    if isinstance(y, torch.Tensor):
        val = y.detach().cpu().numpy()
        if val.size == 1:
            return float(val.item())
        return float(val.reshape(-1)[0])
    else:
        return float(y)

def negamax(model, board: chess.Board, depth: int, alpha: float, beta: float, device='cpu') -> typing.Tuple[float, chess.Move]:
    if depth == 0 or board.is_game_over():
        if board.is_checkmate():
            return (-9999.0, None)
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return (0.0, None)
        val = eval_position(model, board, device=device)
        if board.turn == chess.WHITE:
            return (val, None)
        else:
            return (-val, None)

    best_score = -1e9
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        score_child, _ = negamax(model, board, depth-1, -beta, -alpha, device=device)
        score_child = -score_child
        board.pop()

        if score_child > best_score:
            best_score = score_child
            best_move = move
        alpha = max(alpha, score_child)
        if alpha >= beta:
            break

    return best_score, best_move

def choose_move_negamax(model, board: chess.Board, depth: int = 1, device='cpu') -> typing.Tuple[chess.Move, float]:
    score, move = negamax(model, board, depth, alpha=-1e9, beta=1e9, device=device)
    return move, score

#TODO: play with ai