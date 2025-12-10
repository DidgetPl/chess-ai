import os
import re

import chess
import pygame
from model_loader import list_available_models

PANEL_WIDTH = 300
FONT = None

class GameUI:
    def __init__(self, screen, game):
        global FONT
        self.screen = screen
        self.game = game
        FONT = pygame.font.SysFont(None, 32)

        self.models = list_available_models()
        self.model_index = self.models.index(game.model_name) if game.model_name in self.models else 0

        self.game_time = 0

        self.last_tick = pygame.time.get_ticks()

        self.restart_button = pygame.Rect(650, 400, 190, 50)
        self.model_next_button = pygame.Rect(820, 480, 50, 40)
        self.model_prev_button = pygame.Rect(750, 480, 50, 40)

    def update_timers(self):
        now = pygame.time.get_ticks()
        dt = (now - self.last_tick) / 1000.0
        self.last_tick = now

        if not self.game.game_over_msg:
            self.game_time += dt

    def handle_click(self, pos):
        if self.restart_button.collidepoint(pos):
            self.game_time = 0
            self.game.restart()

        elif self.model_next_button.collidepoint(pos):
            self.model_index = (self.model_index + 1) % len(self.models)
            self.game.set_model(self.models[self.model_index])

        elif self.model_prev_button.collidepoint(pos):
            self.model_index = (self.model_index - 1) % len(self.models)
            self.game.set_model(self.models[self.model_index])

    def prettify_model_name(self, path: str) -> str:
        parts = path.replace("\\", "/").split("/")
        if len(parts) < 2:
            return path

        folder = parts[0]
        filename = parts[-1]

        m = re.search(r"train_loss=([0-9]*\.?[0-9]+)", filename)
        if not m:
            return folder

        train_loss = float(m.group(1))
        train_loss = round(train_loss, 2)

        return f"{folder} (tl = {train_loss:.2f})"


    def draw_panel(self):
        pygame.draw.rect(self.screen, pygame.Color(196,164,132), pygame.Rect(600, 0, PANEL_WIDTH, 600))

        minutes = int(self.game_time // 60)
        seconds = int(self.game_time % 60)
        tenths = int((self.game_time - int(self.game_time)) * 10) if isinstance(self.game_time, float) else 0

        wt = FONT.render(f"Czas gry {minutes}:{seconds:02d}:{tenths}", True, pygame.Color("black"))

        self.screen.blit(wt, (610, 20))

        mc = FONT.render(f"Ruchy: {self.game.move_counter}", True, pygame.Color("black"))
        self.screen.blit(mc, (610, 120))

        pygame.draw.rect(self.screen, pygame.Color(255, 200, 50), self.restart_button)
        rtxt = FONT.render("Zacznij od nowa", True, pygame.Color("black"))
        self.screen.blit(rtxt, (660, 415))

        mtxt = FONT.render("Model:", True, pygame.Color("black"))
        self.screen.blit(mtxt, (610, 480))

        cur = FONT.render(self.prettify_model_name(self.models[self.model_index]), True, pygame.Color("black"))
        self.screen.blit(cur, (610, 530))

        pygame.draw.rect(self.screen, pygame.Color(156,154,122), self.model_next_button)
        pygame.draw.rect(self.screen, pygame.Color(156,154,122), self.model_prev_button)

        self.screen.blit(FONT.render(">", True, pygame.Color("black")), (840, 485))
        self.screen.blit(FONT.render("<", True, pygame.Color("black")), (770, 485))
