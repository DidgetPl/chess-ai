import chess
import pygame
from draw_board_module import draw_board, draw_pieces
from game import ChessGame
from ui import GameUI

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600
SQUARE_SIZE = 600 // 8

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    game = ChessGame(model_name="Model B\\model-epoch=06-train_loss=1.3086.ckpt", ai_depth=3)
    ui = GameUI(screen, game)

    selected_sq = None
    highlight_squares = []

    running = True
    while running:
        screen.fill(pygame.Color("white"))
        ui.update_timers()

        draw_board(screen, highlight_squares, game.last_move_player, game.last_move_ai)
        draw_pieces(screen, game.board)

        ui.draw_panel()

        if game.game_over_msg:
            font = pygame.font.SysFont(None, 50)
            txt = font.render(game.game_over_msg, True, pygame.Color("yellow"))
            rect = txt.get_rect(center=(300, 300))
            screen.blit(txt, rect)
        else:
            if game.board.turn == chess.BLACK:
                game.ai_move()

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                if x > 600:
                    ui.handle_click(event.pos)
                    continue

                c = x // SQUARE_SIZE
                r = y // SQUARE_SIZE
                sq = chess.square(c, 7 - r)

                if selected_sq is None:
                    piece = game.board.piece_at(sq)
                    if piece and piece.color == chess.WHITE:
                        selected_sq = sq
                        highlight_squares = [m.to_square for m in game.board.legal_moves if m.from_square == sq]
                else:
                    move = chess.Move(selected_sq, sq)
                    if game.player_move(move):
                        selected_sq = None
                        highlight_squares = []
                    else:
                        selected_sq = None
                        highlight_squares = []

    pygame.quit()


if __name__ == "__main__":
    main()
