import chess
import pygame
from variables import *

PIECE_IMAGES = {}
pieces = ['wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK']
for p in pieces:
    PIECE_IMAGES[p] = pygame.transform.scale(pygame.image.load(f"./img/{p}.png"), (SQUARE_SIZE, SQUARE_SIZE))

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
