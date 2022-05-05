# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
import carla

from AVR import Utils

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

"""
Contained in HUD, has HUD as member
"""
class KeyboardControl(object):
    def __init__(self, hud):
        self._autopilot_enabled = False
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._hud = hud
        hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            dt_ms = 1/Utils.CarlaFPS*1000
            self._parse_vehicle_keys(pygame.key.get_pressed(), dt_ms)
            self._hud.human_control.steer = self._control.steer
            self._hud.human_control.throttle = self._control.throttle
            self._hud.human_control.brake = self._control.brake
            self._hud.human_control.hand_brake = self._control.hand_brake

            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    print("quit_shortcut")
                    return True
                elif event.key == K_BACKSPACE:
                    print("Next Vehicle")
                    self._hud.next_vehicle()
                elif event.key == K_F1:
                    self._hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    self._hud.help.toggle()
                elif event.key == K_TAB:
                    print("toggle_camera")
                    self._hud.toggle_camera()
                # elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                #     world.next_weather(reverse=True)
                # elif event.key == K_c:
                #     world.next_weather()
                # elif event.key == K_BACKQUOTE:
                #     world.camera_manager.next_sensor()
                # elif event.key > K_0 and event.key <= K_9:
                #     world.camera_manager.set_sensor(event.key - 1 - K_0)
                # elif event.key == K_r:
                #     world.camera_manager.toggle_recording()
                elif event.key == K_q:
                    self._control.reverse = not self._control.reverse
                # elif event.key == K_p:
                #     self._autopilot_enabled = not self._autopilot_enabled
                #     world.vehicle.set_autopilot(self._autopilot_enabled)
                #     world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        # if not self._autopilot_enabled:
        #     self._parse_keys(pygame.key.get_pressed(), clock.get_time())
        #     self._hud._vehicle.apply_control(self._control)


    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        self._control.throttle = 0.6 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 15.0 * 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            if self._steer_cache > 0:
                self._steer_cache = max(self._steer_cache - steer_increment, 0)
            else:
                self._steer_cache = min(self._steer_cache + steer_increment, 0)

        self._steer_cache = min(0.95, max(-0.95, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    # def _parse_keys(self, keys, milliseconds):
    #     self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
    #     steer_increment = 5e-4 * milliseconds
    #     if keys[K_LEFT] or keys[K_a]:
    #         self._steer_cache -= steer_increment
    #     elif keys[K_RIGHT] or keys[K_d]:
    #         self._steer_cache += steer_increment
    #     else:
    #         self._steer_cache = 0.0
    #     self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
    #     self._control.steer = round(self._steer_cache, 1)
    #     self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
    #     self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

