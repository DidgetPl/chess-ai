import time
import typing

import chess
import chess.engine
import numpy as np
import pygame
import torch

from evaluation_model import *
from evaluation_model import EvaluationModel

WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
FPS = 30
PIECE_IMAGES = {}

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

def load_piece_images():
    pieces = ['wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK']
    for p in pieces:
        PIECE_IMAGES[p] = pygame.transform.scale(pygame.image.load(f"img/{p}.png"), (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(screen):
    colors = [pygame.Color(240,217,181), pygame.Color(181,136,99)]
    for r in range(8):
        for c in range(8):
            color = colors[(r+c)%2]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    font = pygame.font.SysFont(None, 20)
    for r in range(8):
        txt = font.render(str(8-r), True, pygame.Color('black'))
        screen.blit(txt, (0, r*SQUARE_SIZE))
    for c in range(8):
        txt = font.render(chr(ord('a')+c), True, pygame.Color('black'))
        screen.blit(txt, ((c+0.5)*SQUARE_SIZE-5, HEIGHT-15))

def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            r, c = divmod(63-square, 8)
            name = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
            screen.blit(PIECE_IMAGES[name], pygame.Rect(c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def play_vs_ai_pygame(model, depth=1, device='cpu'):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    load_piece_images()

    board = chess.Board()
    selected_sq = None
    running = True

    model.eval()
    model.to(device)

    while running:
        draw_board(screen)
        draw_pieces(screen, board)
        pygame.display.flip()
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == chess.WHITE:
                x, y = WIDTH-event.pos[0], event.pos[1]
                c, r = x // SQUARE_SIZE, y // SQUARE_SIZE
                sq = chess.square(c, 7-r)
                if selected_sq is None:
                    selected_sq = sq
                else:
                    move = chess.Move(selected_sq, sq)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_sq = None
                    else:
                        selected_sq = None

        if board.turn == chess.BLACK and not board.is_game_over():
            move, score = choose_move_negamax(model, board, depth=depth, device=device)
            if move:
                board.push(move)

    pygame.quit()

checkpoint_path = "checkpoints\\model-epoch=02-train_loss=3.1472.ckpt"
model = EvaluationModel.load_from_checkpoint(checkpoint_path)
model.eval()
model.to("cpu")

play_vs_ai_pygame(model, depth=2, device='cpu')