/** Render server-integrated particle curves. Null survivors break the streak. */
export function drawFlowCurves(context, flow, transform, theme, layers) {
  const draw = (points, color, dash, particles) => {
    context.save();
    context.lineJoin = "round";
    context.lineCap = "round";
    context.setLineDash(dash);
    context.beginPath();
    let connected = false;
    for (const point of points) {
      if (point === null) { connected = false; continue; }
      const [x, y] = transform.worldToCanvas(...point);
      if (connected) context.lineTo(x, y);
      else context.moveTo(x, y);
      connected = true;
    }
    context.lineWidth = 4.8;
    context.strokeStyle = theme.line.halo;
    context.stroke();
    context.lineWidth = 2.2;
    context.strokeStyle = color;
    context.stroke();
    context.setLineDash([]);
    // Streak samples are simultaneous particles, with no tangent/velocity arrows.
    // Pathlines mark their final particle, including a zero-duration trajectory.
    const markers = particles ? points : points.slice(-1);
    for (const point of markers) {
      if (point === null) continue;
      const [x, y] = transform.worldToCanvas(...point);
      context.beginPath();
      context.arc(x, y, particles ? 3 : 4.5, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      context.lineWidth = 1.5;
      context.strokeStyle = theme.paper;
      context.stroke();
    }
    context.restore();
  };
  if (layers.pathlines) flow.pathlines.forEach((line) => draw(line.points, theme.flow.pathline, [7, 4], false));
  if (layers.streaklines) flow.streaklines.forEach((line) => draw(line.points, theme.flow.streakline, [2, 5], true));
}
