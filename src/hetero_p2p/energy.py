"""Board energy from bracketed sensor samples on the same monotonic clock."""
import math


def integrate_power(rows, start, end, max_gap=3.0, max_command_seconds=1.5):
    """Linear midpoint interpolation; reject gaps/errors rather than extrapolate."""
    if not all(math.isfinite(x) for x in (start, end, max_gap, max_command_seconds)) or end <= start or min(max_gap,max_command_seconds)<=0:
        raise ValueError('invalid energy window or sampling limits')
    points = []
    for row in rows:
        lo, hi = row['start'], row['end']
        if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
            raise ValueError('invalid sensor timestamp')
        if 'error' in row or 'watts' not in row:
            if hi >= start and lo <= end:
                raise ValueError('sensor error overlaps measurement window')
            continue
        watts = row['watts']
        if not math.isfinite(watts) or watts < 0:
            raise ValueError('invalid sensor power')
        points.append(((lo+hi)/2, watts, hi-lo))
    points.sort()
    if len(points)<2 or points[0][0]>start or points[-1][0]<end:
        raise ValueError('sensor samples do not bracket window')
    if any(b[0]<=a[0] for a,b in zip(points,points[1:])):
        raise ValueError('duplicate sensor sample timestamps')
    joules = 0.0
    for (t,w,dt),(u,v,du) in zip(points,points[1:]):
        lo,hi=max(start,t),min(end,u)
        if hi<=lo:
            continue
        if u-t>max_gap or max(dt,du)>max_command_seconds:
            raise ValueError('sensor gap or command duration exceeds limit')
        joules += (hi-lo)*(2*w+(v-w)*(lo+hi-2*t)/(u-t))/2
    return joules
