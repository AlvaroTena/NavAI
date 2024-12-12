from enum import Enum


class ReferenceMode(str, Enum):
    KINEMATIC = "KINEMATIC"
    STATIC = "STATIC"


class ReferenceType:
    class Kinematic(str, Enum):
        RTPPP_GSHARP = "RTPPP_GSHARP"
        RTPPP_PE = "RTPPP_PE"
        RTPPP_MAGIC = "RTPPP_MAGIC"
        SPAN_FILE = "SPAN_FILE"
        RTK_FILE = "RTK_FILE"

    class Static(str, Enum):
        XYZ = "XYZ"
        LLH = "LLH"


SPAN_COLUMNS = [
    "LocalDate",
    "GPSTime",
    "X-ECEF",
    "Y-ECEF",
    "Z-ECEF",
    "Latitude",
    "Longitude",
    "H-Ell",
    "VNorth",
    "VEast",
    "VUp",
    "AccBdyX",
    "AccBdyY",
    "AccBdyZ",
    "GyroDriftX",
    "GyroDriftY",
    "GyroDriftZ",
    "GP",
    "GL",
    "NS",
    "HDOP",
    "VDOP",
    "PDOP",
    "HzSpeed",
    "Heading",
    "ClkCorr",
    "Q",
    "iFlag",
    "AmbStatus",
    "Pitch",
    "Roll",
    "GA",
    "BD",
]
