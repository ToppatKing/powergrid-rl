import pygame
import sys

class GridRenderer:
    def __init__(self, window_size=(900, 700)):
        pygame.init()
        self.width, self.height = window_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Real-Time Power Grid RL Sandbox")
        
        self.font_title = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_text = pygame.font.SysFont("Arial", 16)
        
        # Colors
        self.BG_COLOR = (30, 30, 30)
        self.TEXT_COLOR = (220, 220, 220)
        self.BUS_COLOR = (100, 150, 255)
        
        # Bus coordinates on screen (x, y)
        self.bus_coords = {
            0: (200, 200),  # Top Left: Coal (Slack)
            1: (700, 200),  # Top Right: Gas/Nuclear
            2: (200, 550),  # Bottom Left: Green / Battery
            3: (700, 550)   # Bottom Right: Demand
        }

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def render(self, env, episode, step, ep_reward):
        self.handle_events()
        self.screen.fill(self.BG_COLOR)
        
        # Draw Transmission Lines
        for line in env.grid.lines:
            start_pos = self.bus_coords[line.from_bus]
            end_pos = self.bus_coords[line.to_bus]
            
            # Color logic based on thermal load
            load_pct = abs(line.current_flow) / line.max_flow
            if load_pct > 1.0:
                color = (255, 50, 50)  # Red (Overloaded)
                width = 5
            else:
                # Interpolate from Green to Yellow to Orange
                r = int(min(255, load_pct * 2 * 255))
                g = int(min(255, (1 - load_pct) * 2 * 255))
                color = (r, max(0, g), 0)
                width = max(2, int(load_pct * 5))
                
            pygame.draw.line(self.screen, color, start_pos, end_pos, width)
            
            # Draw flow text in the middle of the line
            mid_x = (start_pos[0] + end_pos[0]) // 2
            mid_y = (start_pos[1] + end_pos[1]) // 2
            flow_text = self.font_text.render(f"{abs(line.current_flow):.1f} / {line.max_flow} MW", True, color)
            self.screen.blit(flow_text, (mid_x - 30, mid_y - 20))

        # Draw Buses
        for bus_id, pos in self.bus_coords.items():
            pygame.draw.circle(self.screen, self.BUS_COLOR, pos, 25)
            pygame.draw.circle(self.screen, (255,255,255), pos, 25, 2)
            lbl = self.font_title.render(f"Bus {bus_id}", True, (0,0,0))
            self.screen.blit(lbl, (pos[0] - 22, pos[1] - 12))

        # Draw Component Statuses
        self._draw_text_box(f"Bus 0: Fossil Base\nCoal-1: {env.generators['Coal-1'].current_mw:.1f} MW\nCoal-2: {env.generators['Coal-2'].current_mw:.1f} MW", (50, 100))
        self._draw_text_box(f"Bus 1: Fast/Nuclear\nGas: {env.generators['Gas'].current_mw:.1f} MW\nNuclear: {env.generators['Nuclear'].current_mw:.1f} MW", (680, 100))
        
        bat_status = "Discharging" if env.battery.current_power_mw > 0 else ("Charging" if env.battery.current_power_mw < 0 else "Idle")
        self._draw_text_box(f"Bus 2: Renewables & Storage\nSolar: {env.solar.current_mw:.1f} MW\nWind: {env.wind.current_mw:.1f} MW\nBattery: {env.battery.current_power_mw:.1f} MW ({bat_status})\nSOC: {env.battery.get_soc()*100:.1f}%", (20, 600))
        
        self._draw_text_box(f"Bus 3: City Center\nDemand: {env.demand.current_load:.1f} MW", (680, 600))

        # Draw Global UI (Top center)
        hz_color = (0, 255, 0) if 59.8 <= env.stability.current_hz <= 60.2 else (255, 0, 0)
        ui_text = [
            f"Time of Day: {env.time_of_day_hours:.1f}h / 24.0h",
            f"Episode: {episode} | Step: {step}",
            f"Reward: {ep_reward:.1f}",
            f"Grid Freq: {env.stability.current_hz:.3f} Hz"
        ]
        
        for i, text in enumerate(ui_text):
            color = hz_color if "Freq" in text else self.TEXT_COLOR
            surface = self.font_title.render(text, True, color)
            self.screen.blit(surface, (350, 20 + i * 25))

        pygame.display.flip()
        
    def _draw_text_box(self, text, pos):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            color = (255, 200, 100) if i == 0 else self.TEXT_COLOR
            font = self.font_title if i == 0 else self.font_text
            surface = font.render(line, True, color)
            self.screen.blit(surface, (pos[0], pos[1] + i * 22))
