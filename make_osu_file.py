import torch


def tensor_to_osu_map(x, y, t, d) -> str:
    lines = []
    slider_start = None
    slider_points = []
    repeats = 1
    cumulative_sum = 0
    slider_active = False

    for i in range(len(x)):
        x_val, y_val, t_val, obj_type = x[i].item(), y[i].item(), (t[i] * 1000).to(torch.int).item(), d[i].item()
        cumulative_sum += t_val

        if obj_type in {1, 3, 4, 5, 6}:
            line = f"{x_val},{y_val},{cumulative_sum},1,0"
            lines.append(line)

        elif obj_type == 2:
            slider_start = (x_val, y_val, t_val)
            slider_points = [(x_val, y_val)]
            repeats = 1
            slider_active = True

        elif obj_type == 8 and slider_active:
            slider_points.append((x_val, y_val))

        elif obj_type == 9 and slider_active:
            repeats += 1

        elif obj_type == 10 and slider_active:
            # End of the slider — finalize
            if slider_start and len(slider_points) >= 1:
                path_type = "B"  # Bezier
                path = path_type + "|" + "|".join(f"{px}:{py}" for px, py in slider_points[1:])  # Skip start point
                pixel_length = 100  # Placeholder; in osu! maps this affects duration
                start_x, start_y, start_t = slider_start
                cumulative_sum += start_t
                line = f"{start_x},{start_y},{cumulative_sum},2,0,{path},{repeats},{pixel_length}"
                lines.append(line)

            # Reset slider state
            slider_start = None
            slider_points = []
            repeats = 1
            slider_active = False

        else:
            # Unrecognized object type, skip
            continue

    return "\n".join(lines)
