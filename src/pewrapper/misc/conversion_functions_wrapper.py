# Value of the constant to convert from degrees to secs
DEG_TO_MIN = 60.0


def getInt(dInput):
    if dInput > 0:
        return int(dInput + 0.5)
    else:
        return int(dInput - 0.5)


def getFloor(dInput):
    if dInput > 0:
        return int(dInput)
    else:
        siResult = int(dInput - 1.0)
        if float(siResult + 1) <= dInput:
            siResult += 1
        return siResult


def convert_deg_to_int_min_sec(deg):
    # Get degree integer part
    val = deg
    deg_int = int(val)

    # Get minute part. Note that minutes are always positive (i.e., the sign is in the degrees part)
    val -= deg_int
    if val < 0:
        val = -val

    val *= DEG_TO_MIN
    deg_min = int(val)

    # Get second part
    val -= deg_min
    val *= DEG_TO_MIN
    deg_sec = val

    return deg_int, deg_min, deg_sec
