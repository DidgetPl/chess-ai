import time
import typing

import chess
import chess.engine
import numpy as np
import pygame
import torch
from evaluation_model import *

WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
FPS = 30
PIECE_IMAGES = {}

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

pygame.display.set_caption("Chess AI")
icon_surface = pygame.image.load("img/wK.png")
pygame.display.set_icon(icon_surface)

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

def draw_game_over(screen, text):
    font = pygame.font.SysFont(None, 30)
    txt = font.render(text, True, pygame.Color('black'), pygame.Color('yellow'))
    rect = txt.get_rect(center=(WIDTH//2, HEIGHT//2))
    screen.blit(txt, rect)

def check_game_over(board):
    if board.is_checkmate():
        return "Szach-mat! " + ("Wygrywa CZARNY" if board.turn == chess.WHITE else "Wygrywa BIAŁY")
    if board.is_stalemate():
        return "Pat – Remis"
    if board.is_insufficient_material():
        return "Remis – Za mało materiału"
    if board.can_claim_threefold_repetition():
        return "Trzykrotne powtórzenie – Remis"
    if board.can_claim_fifty_moves():
        return "50 ruchów bez bicia/pionka – Remis"
    return None

def draw_board(screen, highlight_squares=None, last_move_player=None, last_move_ai=None):
    colors = [pygame.Color(240,217,181), pygame.Color(181,136,99)]
    
    for r in range(8):
        for c in range(8):
            sq = chess.square(c, 7-r)
            color = colors[(r+c)%2]
            rect = pygame.Rect(c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)

            if last_move_player and (sq == last_move_player.from_square or sq == last_move_player.to_square):
                pygame.draw.rect(screen, pygame.Color(120, 255, 120), rect)
            elif last_move_ai and (sq == last_move_ai.from_square or sq == last_move_ai.to_square):
                pygame.draw.rect(screen, pygame.Color(255, 150, 150), rect)
            elif highlight_squares and sq in highlight_squares:
                pygame.draw.rect(screen, pygame.Color(150, 150, 255), rect)
            else:
                pygame.draw.rect(screen, color, rect)

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
            screen.blit(PIECE_IMAGES[name], pygame.Rect(WIDTH - (c+1)*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def play_vs_ai_pygame(model, depth=1, device='cpu'):
    last_move_player = None
    last_move_ai = None
    highlight_squares = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    load_piece_images()

    board = chess.Board()
    selected_sq = None
    running = True
    game_over_message = None

    model.eval()
    model.to(device)




    while running:
        draw_board(screen, highlight_squares, last_move_player, last_move_ai)
        draw_pieces(screen, board)
        if game_over_message:
            draw_game_over(screen, game_over_message)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            continue

        pygame.display.flip()
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == chess.WHITE:
                x, y = event.pos[0], event.pos[1]
                c, r = x // SQUARE_SIZE, y // SQUARE_SIZE
                sq = chess.square(c, 7-r)

                if selected_sq is None:
                    piece = board.piece_at(sq)

                    if piece and piece.color == chess.WHITE:
                        selected_sq = sq

                        highlight_squares = []
                        for m in board.legal_moves:
                            if m.from_square == sq:
                                highlight_squares.append(m.to_square)
                    else:
                        selected_sq = None
                        highlight_squares = []

                else:
                    move = None

                    piece = board.piece_at(selected_sq)
                    if piece and piece.piece_type == chess.PAWN:
                        target_rank = chess.square_rank(sq)
                        if (piece.color == chess.WHITE and target_rank == 7) or (piece.color == chess.BLACK and target_rank == 0):

                            def choose_promotion_piece(color):
                                candidates = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                                for pt in candidates:
                                    if len(board.pieces(pt, color)) < len(board.pieces(pt, not color)):
                                        return pt
                                return chess.QUEEN

                            promotion_piece = choose_promotion_piece(piece.color)
                            move = chess.Move(selected_sq, sq, promotion=promotion_piece)

                    if move is None:
                        move = chess.Move(selected_sq, sq)

                    if move in board.legal_moves:
                        board.push(move)
                        last_move_player = move
                        highlight_squares = []
                        game_over_message = check_game_over(board)
                        selected_sq = None
                    else:
                        print("wait, that's illegal")
                        selected_sq = None


        if board.turn == chess.BLACK and not board.is_game_over():
            move, score = choose_move_negamax(model, board, depth=depth, device=device)
            if move:
                board.push(move)
                last_move_ai = move
                game_over_message = check_game_over(board)

    pygame.quit()

checkpoint_path = ".\\training\\checkpoints\\Model B\\model-epoch=06-train_loss=1.3086.ckpt"
model = EvaluationModel.load_from_checkpoint(checkpoint_path)
model.eval()
model.to("cpu")

play_vs_ai_pygame(model, depth=2, device='cpu')