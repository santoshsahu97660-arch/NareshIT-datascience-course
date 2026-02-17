import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Define constants
WIDTH, HEIGHT = 640, 480
FPS = 60
BRUSH_SIZE = 8

# Set up the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw with Your Hand!")

# Create transparent drawing surface
drawing_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

# Colors
BRUSH_COLOR = (0, 255, 0)

# Hand Tracking Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


def detect_hand(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]
    return None


def hand_to_screen(hand_landmarks):
    x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WIDTH)
    y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * HEIGHT)
    return x, y


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    clock = pygame.time.Clock()
    last_pos = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        hand_landmarks = detect_hand(frame)

        if hand_landmarks:
            x, y = hand_to_screen(hand_landmarks)

            if last_pos:
                pygame.draw.line(drawing_surface,
                                 BRUSH_COLOR,
                                 last_pos,
                                 (x, y),
                                 BRUSH_SIZE)

            last_pos = (x, y)
        else:
            last_pos = None

        # Convert frame for pygame (FIX ROTATION)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame_surface = pygame.surfarray.make_surface(frame)

        # Draw background first
        screen.blit(frame_surface, (0, 0))

        # Then draw lines on top
        screen.blit(drawing_surface, (0, 0))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                cv2.destroyAllWindows()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cap.release()
                    pygame.quit()
                    cv2.destroyAllWindows()
                    return
                if event.key == pygame.K_c:
                    drawing_surface.fill((0, 0, 0, 0))  # Clear drawing

        clock.tick(FPS)


if __name__ == "__main__":
    main()
