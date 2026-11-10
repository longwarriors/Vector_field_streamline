"""Shared browser fixture data, with no pytest fixture registration."""

def _browser_presets() -> list[dict[str, object]]:
    editable = {"exclusive_minimum": 0.322, "unit": "m"}
    return [
        {
            "id": "electric_dipole",
            "label": "电偶极子",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "electric_quadrupole",
            "label": "电四极子",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "electric_hexagon",
            "label": "六个正电荷",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "electric_hexagon_alternating",
            "label": "三对交替电荷",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "magnetic_dipole",
            "label": "磁偶极子",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "halbach_array",
            "label": "Halbach 阵列",
            "description": "fixture",
            "source_separation": editable,
        },
        {"id": "current_loop", "label": "圆形电流线圈", "description": "fixture"},
        {"id": "straight_wire", "label": "长直导线", "description": "fixture"},
        {"id": "parallel_wires", "label": "同向双导线", "description": "fixture"},
        {"id": "antiparallel_wires", "label": "反向双导线", "description": "fixture"},
        {"id": "solenoid", "label": "有限螺线管", "description": "fixture"},
        {"id": "helmholtz_coils", "label": "Helmholtz 线圈", "description": "fixture"},
        {"id": "anti_helmholtz_coils", "label": "反 Helmholtz 线圈", "description": "fixture"},
        {"id": "charged_ring", "label": "带电圆环", "description": "fixture"},
        {"id": "dielectric_sphere", "label": "介质球", "description": "fixture"},
        {"id": "conducting_sphere", "label": "导体球", "description": "fixture"},
        {"id": "uniform", "label": "匀强场", "description": "fixture"},
    ]
