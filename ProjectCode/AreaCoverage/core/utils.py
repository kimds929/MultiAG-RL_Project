# File: utils.py
import numpy as np
import collections
import cv2
import matplotlib.pyplot as plt
import time

def pattern_walls(name: str, grid_size_y: int, grid_size_x: int) -> np.ndarray:
    H, W = grid_size_y, grid_size_x
    M = np.zeros((H, W), dtype=bool)
    if name == "bridge":
        t = max(1, H // 5)
        wL, wR = W // 4, W - W // 4
        M[t:2*t, wL:wR] = True
        M[H-2*t:H-t, wL:wR] = True
    elif name == "cliff":
        M[H//2:, W//4:W-W//4] = True
    elif name == "zigzag":
        margin = max(1, H // 5)
        M[margin, :W-margin] = True
        M[H-margin-1, margin:] = True
        if H > 7:
            mid = H // 2
            M[mid, margin:W-margin] = True
    else:
        raise ValueError(f"Unknown pattern: {name!r}")
    return M


def is_free_connected(walls: np.ndarray) -> bool:
    H, W = walls.shape
    seen = np.zeros_like(walls, dtype=bool)
    start = None
    for i in range(H):
        for j in range(W):
            if not walls[i, j]:
                start = (i, j)
                break
        if start:
            break
    q = collections.deque([start])
    seen[start] = True
    cnt = 1
    while q:
        y, x = q.popleft()
        for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                cnt += 1
                q.append((ny, nx))
    return cnt == (walls.size - walls.sum())


def sample_random_walls(rng: np.random.RandomState, grid_size_y: int, grid_size_x: int, num_obstacles: int=None) -> np.ndarray:
    total = grid_size_y * grid_size_x
    num = num_obstacles or int(0.25 * total)
    while True:
        idx = rng.choice(total, size=num, replace=False)
        ys, xs = np.divmod(idx, grid_size_x)
        M = np.zeros((grid_size_y, grid_size_x), dtype=bool)
        M[ys, xs] = True
        if is_free_connected(M):
            return M


def render_grid(step: int, grid_size_x: int, grid_size_y: int, walls: np.ndarray,
                coverage: np.ndarray, coverage_owner: np.ndarray, agent_positions: list,
                agent_directions: list, mode: str = "cv2", animate: bool = False,
                info=None):
    cell_size, border = 30, 2
    header_height = 40  # <-- 여백 추가
    img_h = grid_size_y * cell_size + header_height
    img_w = grid_size_x * cell_size
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    if step is not None:
        cv2.putText(img, f"Step: {step}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    colors = [(255, 0, 0), (0, 0, 255), (0, 150, 0), (150, 0, 150), (0, 150, 150)]
    for y in range(grid_size_y):
        for x in range(grid_size_x):
            tl = (x * cell_size + border, y * cell_size + border + header_height)
            br = ((x + 1) * cell_size - border, (y + 1) * cell_size - border + header_height)
            if info[y,x] == 9:
                cv2.rectangle(img, tl, br, (100, 100, 100), -1)
                text = '?'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = tl[0] + (br[0] - tl[0] - text_size[0]) // 2
                text_y = tl[1] + (br[1] - tl[1] + text_size[1]) // 2
                cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            elif walls[y, x]:
                cv2.rectangle(img, tl, br, (0, 0, 0), -1)
            elif coverage[y, x]:
                clr = colors[coverage_owner[y, x] % len(colors)]
                cv2.rectangle(img, tl, br, clr, -1)
            else:
                cv2.rectangle(img, tl, br, (240, 240, 240), -1)
    
    for i, (y, x) in enumerate(agent_positions):
        # (action) 0 : stop(·), 1: up(↑), 2: down(↓), 3: left(←), 4: right(→)
        center = (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2 + header_height)
        r = cell_size // 3
        cv2.circle(img, center, r, tuple(ch * 0.7 for ch in colors[i]), -1)
        cv2.putText(img, str(i), (center[0]-7, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        off = r + 4

        d = agent_directions[i]
        if d == 0:
            tip = (center[0]+off, center[1])
            left = (center[0]+off-7, center[1]-5)
            right = (center[0]+off-7, center[1]+5)
        elif d == 1:
            tip = (center[0], center[1]+off)
            left = (center[0]-5, center[1]+off-7)
            right = (center[0]+5, center[1]+off-7)
        elif d == 2:
            tip = (center[0]-off, center[1])
            left = (center[0]-off+7, center[1]-5)
            right = (center[0]-off+7, center[1]+5)
        elif d == 3:
            tip = (center[0], center[1]-off)
            left = (center[0]-5, center[1]-off+7)
            right = (center[0]+5, center[1]-off+7)
        triangle = np.array([tip, left, right], dtype=np.int32)
        cv2.fillPoly(img, [triangle], (0, 255, 255))

    if mode == "cv2":
        cv2.imshow("Area Coverage", img)
        if animate and cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return img, True
    elif mode == "pyplot":
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.title(f"Area Coverage {step}" if step is not None else "Area Coverage")
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.close()
    
    return img, False