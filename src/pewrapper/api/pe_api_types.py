import ctypes as ct
import math
from typing import Union

from pewrapper.api import Log_Handle, LogCategoryPE, Signal_Obs, f_handle_bin_ephem_NVM
from pewrapper.common.cwrapper import C_Enum
from pewrapper.types import constants
from pewrapper.types.constants import SECURITY

INPUT_IMU_DATA_BUFFER_SIZE = 25
INPUT_WHEEL_SPEED_DATA_BUFFER_SIZE = 25


class NominalState:
    STATE_0_INIT = 0
    STATE_1_LANE_PRECISE = 1
    STATE_2_ROAD_PRECISE = 2
    STATE_3_1_SHORT_DR = 3
    STATE_3_2_LONG_DR = 4
    STATE_3_3_UNLIMITED_DR = 5
    STATE_4_1_DEGRADED_ACCURACY = 6
    STATE_4_2_CS_OUTAGE = 7
    STATE_4_3_EPHEM_ONLY = 8
    STATE_4_4_DEGRADED_DR = 9
    STATE_255_NO_SOLUTION = 10


class DTCActionType(C_Enum):
    ActionPass = 0  # DTC is ok
    ActionFailed = 1  # DTC is raised after the debouncing time
    InstantFailed = 2  # DTC is raised immediately


class DTC_ID_List(C_Enum):
    DTC_MIN = 0
    TELEM_SERVER_DOWN_TIMEOUT = 1
    TELEM_INVALID_SIGNATURE = 2
    CORR_SRVC_INVALID_E2E_STATUS = 3
    GNSS_SERVER_DOWN_TIMEOUT = 4
    GNSS_INVALID_E2E_STATUS = 5
    JAMMING_DETECTION = 6
    SPOOFING_DETECTION = 7
    PE_ALG_ERROR = 8
    INPUT_SENSOR_ERROR = 9
    INVALID_GNSS_RAW_DATA = 10
    INVALID_CORR_SRVC_RAW_DATA = 11
    SAFETY_CONDITION = 12
    DTC_MAX = 13


class SensorQualifier(C_Enum):
    N_A = 0
    VALID = 1
    INVALID = 2


############# IMU_Measurements ###############
class IMU_Measurements(ct.Structure):
    ### ------------------------------------------------------------------------------------###
    ###   IMU's reference system:														    ###
    ### 		x -> longitudinal axis of the car, forwards									###
    ###			z -> vertical axis, downwards												###
    ###			y -> forming a right handed trihedron										###
    ###																						###
    ###			acc = (longitudinalAcceleration, lateralAcceleration, verticalAcceleration)	###
    ###			vel = (rollVelocity,pitchVelocity,yawVelocityVehicle)						###
    ### ----------------------------------------------------------------------------------- ###
    _fields_ = [
        ("sync_ns", ct.c_uint32),
        ("sync_sec", ct.c_double),
        ("linearAcceleration", ct.c_double * 3),
        ("angularVelocity", ct.c_double * 3),
        ("qualifierVelAcc", ct.c_uint8),
    ]

    def __init__(self):
        self.sync_ns = 0
        self.sync_sec = 0.0
        # longitudinalAcceleration, lateralAcceleration, verticalAcceleration
        self.linearAcceleration = (ct.c_double * 3)()
        # rollVelocity, pitchVelocity, yawVelocityVehicle
        self.angularVelocity = (ct.c_double * 3)()
        self.qualifierVelAcc = SensorQualifier.N_A


############# ODO_Measurements ###############
class WheelSpeedData(ct.Structure):
    _fields_ = [
        ("crcActualValueRPMWheel2", ct.c_uint16),
        ("actualValuesRPMWheel", ct.c_double * 4),
        ("timestampActualValueRPMWheel", ct.c_uint32),
        ("sync_sec", ct.c_double),
        ("qualifierActualValuesRPMWheel", ct.c_uint8),
    ]

    def __init__(self):
        # CRC checksum for the extended standard protection with application CRC according to application profile 5 E2E Autosar protection.
        self.crcActualValueRPMWheel2 = 0
        # Measured speed of four wheel: left rear wheel, right rear wheel, left front wheel and right front wheel.
        self.actualValuesRPMWheel = (ct.c_double * 4)()
        # Timestamp.
        self.timestampActualValueRPMWheel = 0
        # UBX A9 sync sec.
        self.sync_sec = 0.0
        # Status qualifier for the four signals.
        self.qualifierActualValuesRPMWheel = SensorQualifier.N_A


##### PE OUTPUT STR ######
class UsedSatellites(ct.Structure):
    _fields_ = [
        ("status", ct.c_bool),
        ("fixed", ct.c_bool),
        ("only_one_freq", ct.c_bool),
        ("PRN", ct.c_int32),
        ("elevation", ct.c_double),
        ("azimuth", ct.c_double),
        ("SNR_1", ct.c_double),
        ("SNR_2", ct.c_double),
        ("CodeResidual_1", ct.c_double),
        ("CodeResidual_2", ct.c_double),
        ("signal_id_1", ct.c_uint32),
        ("signal_id_2", ct.c_uint32),
    ]

    def __init__(self):
        self.status = False
        self.fixed = False
        self.only_one_freq = False
        self.PRN = 0
        self.elevation = 0.0  # radians
        self.azimuth = 0.0  # radians
        self.SNR_1 = 0.0  # dBs
        self.SNR_2 = 0.0  # dBs
        self.CodeResidual_1 = 0.0  # meters
        self.CodeResidual_2 = 0.0  # meters
        self.signal_id_1 = Signal_Obs.SIG_UNKNOWN
        self.signal_id_2 = Signal_Obs.SIG_UNKNOWN


class ProtectionLevel_pos(ct.Structure):
    _fields_ = [
        ("along", ct.c_float),
        ("cross", ct.c_float),
        ("vertical", ct.c_float),
        ("heading", ct.c_double),
        ("horizontal_flag", ct.c_bool),
        ("vertical_flag", ct.c_bool),
        ("heading_flag", ct.c_bool),
    ]

    def __init__(self):
        self.along = 0.0  # meters
        self.cross = 0.0  # meters
        self.vertical = 0.0  # meters
        self.heading = 0.0  # degrees
        self.horizontal_flag = False
        self.vertical_flag = False
        self.heading_flag = False


class SystemStatus(C_Enum):
    NA_SYSTEM_STATUS = 0
    POSITION_FIX = 1
    POSITION_FLOAT_REGIONAL = 2
    POSITION_FLOAT_GLOBAL = 3
    IMU_ONLY = 4
    IMU_NA = 5
    SPP = 6
    SPP_IMU_ONLY = 7
    SPP_IMU_NA = 8
    PVT = 9


class Latitude_Direction(C_Enum):
    NORTH = 0
    SOUTH = 1


class Longitude_Direction(C_Enum):
    EAST = 0
    WEST = 1


class gnssReferencePoint(C_Enum):
    L1E1_ANTENNA_PHASE_CENTER = 0
    REAR_AXLE_CENTER = 1
    VEHICLE_CENTER_SURFACE = 2


class Event(C_Enum):
    FORCE_FATAL_ERROR = 0
    FORCE_JAMMING = 1
    FORCE_SPOOFING = 2


class SafeStateMachineSignal(C_Enum):
    FATAL_ERROR = 0
    RESET_FILTER = 1
    INIT_MON = 2
    CHECK_FAIL_ALGO = 3
    CHECK_OK_ALGO = 4
    NO_SOLUTION = 5
    NON_SAFE_CONDITIONS = 6
    SAFE_CONDITIONS = 7
    START_SIGNAL = 8
    TERMINATE_SIGNAL = 9


class SafeState(C_Enum):
    INACTIVE_STATE = 0
    INIT_STATE = 1
    NO_SOLUTION_STATE = 2
    INIT_MON_STATE = 3
    VALID_SOLUTION_STATE = 4
    SAFE_SOLUTION_STATE = 5
    ERROR_STATE = 6


class SensorStatus(C_Enum):
    SYSTEM_OK = 0
    IMU_FAILURE = 1
    ME_FAILURE = 2
    IMU_ME_FAILURE = 3


PE_imu_odo_out_arrays_size = 3
PE_Solution_Info_used_sats_arrays_size = 40
PE_Solution_Info_used_sats_arrays_size_beidou = 63


class PE_Output_str(ct.Structure):
    class PE_pos_out(ct.Structure):
        _fields_ = [
            ("pos_out", ct.c_double * 3),
            ("latitude", ct.c_double),
            ("longitude", ct.c_double),
            ("height", ct.c_double),
            ("altitude", ct.c_double),
            ("latitude_direction", ct.c_uint32),
            ("longitude_direction", ct.c_uint32),
            ("latitude_sigma", ct.c_float),
            ("longitude_sigma", ct.c_float),
            ("horizontal_sigma", ct.c_float),
            ("height_sigma", ct.c_float),
            ("altitude_sigma", ct.c_float),
            ("heading", ct.c_double),
            ("heading_flag", ct.c_bool),
            ("heading_sigma", ct.c_float),
            ("along_track_sigma", ct.c_float),
            ("cross_track_sigma", ct.c_float),
        ]

        def __init__(self):
            self.pos_out = (ct.c_double * 3)()  # X,Y,Z components in meters
            self.latitude = 0.0  # degrees
            self.longitude = 0.0  # degrees
            self.height = 0.0  # meters
            self.altitude = 0.0  # meters
            self.latitude_direction = Latitude_Direction.NORTH
            self.longitude_direction = Longitude_Direction.EAST
            self.latitude_sigma = 0.0
            self.longitude_sigma = 0.0
            self.horizontal_sigma = 0.0
            self.height_sigma = 0.0
            self.altitude_sigma = 0.0
            self.heading = 0.0  # degrees
            self.heading_flag = False  # Validity of heading
            self.heading_sigma = 0.0
            self.along_track_sigma = 0.0
            self.cross_track_sigma = 0.0

    class PE_vel_out(ct.Structure):
        _fields_ = [
            ("vel_out", ct.c_double * 3),
            ("velocityHorizontal", ct.c_double),
            ("velocityVertical", ct.c_double),
            ("TrackAngle", ct.c_double),
            ("velocityModule_knots", ct.c_double),
            ("velocityModule_kmh", ct.c_double),
            ("horizontal_sigma", ct.c_double),
            ("vertical_sigma", ct.c_double),
            ("track_angle_sigma", ct.c_double),
        ]

        def __init__(self):
            self.vel_out = (ct.c_double * 3)()  # X,Y,Z components in meters/seconds
            self.velocityHorizontal = 0.0  # meters/seconds
            self.velocityVertical = 0.0  # meters/seconds
            self.TrackAngle = 0.0  # degrees
            self.velocityModule_knots = 0.0  # knots
            self.velocityModule_kmh = 0.0  # kilometers / hour
            self.horizontal_sigma = 0.0
            self.vertical_sigma = 0.0
            self.track_angle_sigma = 0.0

    class PE_Receiver_Clock_out(ct.Structure):
        _fields_ = [("ClockBias", ct.c_double), ("ClockDrift", ct.c_double)]

        def __init__(self):
            self.ClockBias = 0.0  # meters
            self.ClockDrift = 0.0  # meters

    class PE_integrity_pos(ct.Structure):
        _fields_ = [
            ("TIR_10_3", ProtectionLevel_pos),
            ("TIR_10_4", ProtectionLevel_pos),
            ("TIR_10_5", ProtectionLevel_pos),
            ("TIR_10_6", ProtectionLevel_pos),
            ("TIR_0_003", ProtectionLevel_pos),
        ]

        def __init__(self):
            self.TIR_10_3 = ProtectionLevel_pos()
            self.TIR_10_4 = ProtectionLevel_pos()
            self.TIR_10_5 = ProtectionLevel_pos()
            self.TIR_10_6 = ProtectionLevel_pos()
            self.TIR_0_003 = ProtectionLevel_pos()

    class PE_integrity_vel(ct.Structure):
        _fields_ = [
            ("PL_horizontal_velocity", ct.c_double),
            ("PL_vertical_velocity", ct.c_double),
            ("PL_courseangle", ct.c_double),
            ("PL_horizontal_velocity_flag", ct.c_bool),
            ("PL_vertical_velocity_flag", ct.c_bool),
            ("PL_courseangle_flag", ct.c_bool),
        ]

        def __init__(self):
            self.PL_horizontal_velocity = 0.0
            self.PL_vertical_velocity = 0.0
            self.PL_courseangle = 0.0
            self.PL_horizontal_velocity_flag = False
            self.PL_vertical_velocity_flag = False
            self.PL_courseangle_flag = False

    class PE_Solution_info(ct.Structure):
        _fields_ = [
            ("SSM_Signal", ct.c_uint32),
            ("sensorStatus", ct.c_uint32),
            ("systemStatus", ct.c_uint32),
            ("nominalState", ct.c_uint32),
            ("PDOP", ct.c_double),
            ("HDOP", ct.c_double),
            ("VDOP", ct.c_double),
            ("total_number_SVS_visible", ct.c_uint32),
            ("SVG_visible", ct.c_uint32),
            ("SVE_visible", ct.c_uint32),
            ("total_used_sats_fixed", ct.c_uint32),
            ("total_used_sats_float", ct.c_uint32),
            ("total_used_sats_float_one_freq", ct.c_uint32),
            (
                "used_sats_info_gps",
                UsedSatellites * PE_Solution_Info_used_sats_arrays_size,
            ),
            (
                "used_sats_info_gal",
                UsedSatellites * PE_Solution_Info_used_sats_arrays_size,
            ),
            (
                "used_sats_info_bds",
                UsedSatellites * PE_Solution_Info_used_sats_arrays_size_beidou,
            ),
            ("gnssReferencePoint", ct.c_uint32),
            ("antennaAxleOffsetX", ct.c_double),
            ("antennaAxleOffsetY", ct.c_double),
            ("antennaAxleOffsetZ", ct.c_double),
            ("vel_core_validity", ct.c_bool),
            ("SSM_Signal_Vel", ct.c_uint32),
            ("correctionAge", ct.c_double),
            ("timeWithoutSatelliteSignals", ct.c_double),
        ]
        if SECURITY:
            _fields_.extend(
                [("serial_license_valid", ct.c_bool), ("time_license_valid", ct.c_bool)]
            )

        def __init__(self):
            self.SSM_Signal = SafeStateMachineSignal.NO_SOLUTION
            self.sensorStatus = SensorStatus.SYSTEM_OK
            self.systemStatus = SystemStatus.NA_SYSTEM_STATUS
            self.nominalState = NominalState.STATE_255_NO_SOLUTION
            self.PDOP = 0.0
            self.HDOP = 0.0
            self.VDOP = 0.0
            self.total_number_SVS_visible = 0
            self.SVG_visible = 0
            self.SVE_visible = 0
            self.total_used_sats_fixed = 0
            self.total_used_sats_float = 0
            self.total_used_sats_float_one_freq = 0
            self.used_sats_info_gps = (
                UsedSatellites * PE_Solution_Info_used_sats_arrays_size
            )()
            self.used_sats_info_gal = (
                UsedSatellites * PE_Solution_Info_used_sats_arrays_size
            )()
            self.used_sats_info_bds = (
                UsedSatellites * PE_Solution_Info_used_sats_arrays_size_beidou
            )()
            self.gnssReferencePoint = gnssReferencePoint.L1E1_ANTENNA_PHASE_CENTER
            self.antennaAxleOffsetX = 0.0
            self.antennaAxleOffsetY = 0.0
            self.antennaAxleOffsetZ = 0.0
            self.vel_core_validity = False
            self.SSM_Signal_Vel = SafeStateMachineSignal.NO_SOLUTION
            self.correctionAge = 0.0
            self.timeWithoutSatelliteSignals = 0.0

            if SECURITY:
                self.serial_license_valid = True
                self.time_license_valid = True

    class PE_imu_odo_out(ct.Structure):
        _fields_ = [
            ("acc_bias", ct.c_double * PE_imu_odo_out_arrays_size),
            ("gyr_bias", ct.c_double * PE_imu_odo_out_arrays_size),
            ("att", ct.c_double * PE_imu_odo_out_arrays_size),
            ("scale_factor", ct.c_double),
        ]

        def __init__(self):
            self.acc_bias = (ct.c_double * PE_imu_odo_out_arrays_size)()
            self.gyr_bias = (ct.c_double * PE_imu_odo_out_arrays_size)()
            self.att = (ct.c_double * PE_imu_odo_out_arrays_size)()
            self.scale_factor = 0.0

    _fields_ = [
        ("timestamp_week", ct.c_uint32),
        ("timestamp_second", ct.c_double),
        ("GM_time", ct.c_double),
        ("pe_pos_out", PE_pos_out),
        ("pe_vel_out", PE_vel_out),
        ("pe_receiver_clock_out", PE_Receiver_Clock_out),
        ("pe_integrity_pos", PE_integrity_pos),
        ("pe_integrity_vel", PE_integrity_vel),
        ("pe_solution_info", PE_Solution_info),
        ("pe_imu_odo_out", PE_imu_odo_out),
    ]

    def __init__(self):
        self.timestamp_week = 0
        self.timestamp_second = 0.0
        self.GM_time = 0.0
        self.pe_pos_out = PE_Output_str.PE_pos_out()
        self.pe_vel_out = PE_Output_str.PE_vel_out()
        self.pe_receiver_clock_out = PE_Output_str.PE_Receiver_Clock_out()
        self.pe_integrity_pos = PE_Output_str.PE_integrity_pos()
        self.pe_integrity_vel = PE_Output_str.PE_integrity_vel()
        self.pe_solution_info = PE_Output_str.PE_Solution_info()
        self.pe_imu_odo_out = PE_Output_str.PE_imu_odo_out()


class type_receier(C_Enum):
    E_TESEO = 0
    E_UBLOX = 1
    E_UBLOX_A9 = 2
    E_MONITOR = 3
    UNKNOWN = 4


class doppler_sign(C_Enum):
    POSITIVE = 0
    NEGATIVE = 1
    UNKNOWN_SIGN = 2


class GNSSProtocol(C_Enum):
    PROTOCOL_UNKNOWN = 0
    RTCM = 1
    UBX = 2
    SBF = 3


class GalCorrectionDataType(C_Enum):
    CORRECTION_DATA_UNKNOWN = 0
    F_NAV = 1
    I_NAV = 2


class GM_Time(ct.Structure):
    _fields_ = [
        ("value", ct.c_double),
        ("valid", ct.c_bool),
    ]

    def __init__(self):
        self.value = 0.0  # in seconds
        self.valid = False


class API_Time(ct.Structure):
    _fields_ = [
        ("valid", ct.c_bool),
        ("week", ct.c_uint32),
        ("second", ct.c_double),
    ]

    def __init__(self):
        self.valid = False
        self.week = 0
        self.second = 0.0


f_handle_ReportDTCStatus = ct.CFUNCTYPE(
    ct.c_bool,  # return
    ct.c_uint32,  # DTC_ID_List DTC
    ct.c_uint32,  # const DTCActionType DTCStatus
)


class Configuration_info(ct.Structure):
    _fields_ = [
        ("parallelize", ct.c_bool),
        ("max_num_pe_log_files", ct.c_int32),
        ("max_num_wrapper_data_files", ct.c_int32),
        ("log_path", ct.c_char_p),
        ("tracing_config_file", ct.c_char_p),
        ("algo_profile", ct.c_uint32),
        ("doppler_sign", ct.c_uint32),
        ("use_velCore", ct.c_bool),
        ("require_e2e_msg", ct.c_bool),
        ("require_timesync", ct.c_bool),
        ("use_global_iono", ct.c_bool),
        ("use_iono", ct.c_bool),
        ("cn0_threshold_signal_1", ct.c_double),
        ("cn0_threshold_signal_2", ct.c_double),
        ("Signal_1_GAL", ct.c_uint32),
        ("Signal_2_GAL", ct.c_uint32),
        ("Signal_1_GPS", ct.c_uint32),
        ("Signal_2_GPS", ct.c_uint32),
        ("Signal_1_BDS", ct.c_uint32),
        ("Signal_2_BDS", ct.c_uint32),
        ("gnssProtocol", ct.c_uint32),
        ("galCorrectionDataType", ct.c_uint32),
        ("use_gps", ct.c_bool),
        ("use_gal", ct.c_bool),
        ("use_bds", ct.c_bool),
        ("use_AI_multipath", ct.c_bool),
        ("use_imu", ct.c_bool),
        ("use_wt", ct.c_bool),
        ("use_Integrity", ct.c_bool),
        ("require_CS_integrity", ct.c_bool),
        ("use_qm_variant", ct.c_bool),
        ("perform_js_checks", ct.c_bool),
        ("imu_2_antenna_lever_arm", ct.c_double * 3),
        ("antenna_2_rear_axle", ct.c_double * 3),
        ("RearWheelDistance", ct.c_double),
        ("WheelDiameter", ct.c_double),
        ("tile_hysteresis_delta_km", ct.c_double),
        ("doppler_available", ct.c_bool),
        ("max_age_of_corrections", ct.c_double),
        ("binEphem_handle", f_handle_bin_ephem_NVM),
        ("log_handle", Log_Handle),
        ("log_category", ct.c_uint32),
        ("ReportDTCStatus", f_handle_ReportDTCStatus),
    ]
    if SECURITY:
        _fields_.extend([("license_file_path", ct.c_char_p)])

    def __init__(
        self,
        binEphem_handle,
        log_handle,
        ReportDTCStatus,
    ):
        self.parallelize = False
        self.max_num_pe_log_files = 48
        self.max_num_wrapper_data_files = 48

        self.log_path = "./".encode("utf-8")
        self.tracing_config_file = "./".encode("utf-8")

        self.algo_profile = ct.c_uint32(type_receier.E_TESEO)
        self.doppler_sign = ct.c_uint32(doppler_sign.UNKNOWN_SIGN)

        self.use_velCore = True
        self.require_e2e_msg = True  # UBX SEC-CRC
        self.require_timesync = False

        self.use_global_iono = False
        self.use_iono = True

        self.cn0_threshold_signal_1 = 32.0
        self.cn0_threshold_signal_2 = 32.0

        self.Signal_1_GAL = ct.c_uint32(Signal_Obs.GAL_E1_BC)
        self.Signal_2_GAL = ct.c_uint32(Signal_Obs.GAL_E5B_Q)
        self.Signal_1_GPS = ct.c_uint32(Signal_Obs.GPS_L1_C)
        self.Signal_2_GPS = ct.c_uint32(Signal_Obs.GPS_L2_CM)
        self.Signal_1_BDS = ct.c_uint32(Signal_Obs.BDS_B1I_D1)
        self.Signal_2_BDS = ct.c_uint32(Signal_Obs.BDS_B2A_P)

        self.gnssProtocol = ct.c_uint32(GNSSProtocol.PROTOCOL_UNKNOWN)
        self.galCorrectionDataType = ct.c_uint32(GalCorrectionDataType.F_NAV)

        self.use_gps = True
        self.use_gal = True
        self.use_bds = False

        self.use_AI_multipath = False
        self.use_imu = True
        self.use_wt = False
        self.use_Integrity = False
        self.require_CS_integrity = False
        self.use_qm_variant = False
        self.perform_js_checks = True

        self.imu_2_antenna_lever_arm = (
            ct.c_double * 3
        )()  # ISO8855 axes with components x,y,z in meters
        self.antenna_2_rear_axle = (
            ct.c_double * 3
        )()  # ISO8855 axes with components x,y,z in meters

        self.RearWheelDistance = 0.0  # in meters
        self.WheelDiameter = 0.0  # in meters

        self.tile_hysteresis_delta_km = 20.0
        self.doppler_available = True

        if SECURITY:
            self.license_file_path = "./".encode("utf-8")

        self.max_age_of_corrections = 18000.0

        self.binEphem_handle = binEphem_handle
        self.log_handle = log_handle
        self.log_category = ct.c_uint32(LogCategoryPE.NONE)
        self.ReportDTCStatus = ReportDTCStatus


Input_IMU_Measurements = IMU_Measurements * INPUT_IMU_DATA_BUFFER_SIZE
Input_WheelSpeed_Measurements = WheelSpeedData * INPUT_WHEEL_SPEED_DATA_BUFFER_SIZE


class PE_FeaturesMultipathAI(ct.Structure):
    _fields_ = [
        ("usable", ct.c_bool),
        ("timestamp_week", ct.c_uint32),
        ("timestamp_second", ct.c_double),
        ("sat_id", ct.c_int32),
        ("constel", ct.c_uint32),
        ("freq", ct.c_double),
        ("code", ct.c_double),
        ("phase", ct.c_double),
        ("doppler", ct.c_double),
        ("snr", ct.c_double),
        ("elevation", ct.c_double),
        ("residual", ct.c_double),
        ("iono", ct.c_double),
        ("delta_cmc", ct.c_double),
        ("crc", ct.c_double),
        ("delta_time", ct.c_double),
    ]

    def __init__(self):
        self.usable = False
        self.timestamp_week = 0
        self.timestamp_second = 0.0
        self.sat_id = -1
        self.constel = constants.E_CONSTELLATION.E_CONSTEL_NONE
        self.freq = 0.0
        self.code = 0.0
        self.phase = 0.0
        self.doppler = 0.0
        self.snr = 0.0
        self.elevation = 0.0
        self.residual = 0.0
        self.iono = 0.0
        self.delta_cmc = math.nan
        self.crc = math.nan
        self.delta_time = 0.0


class PE_PredictionsAI(ct.Structure):
    _fields_ = [
        ("timestamp_week", ct.c_uint32),
        ("timestamp_second", ct.c_double),
        ("constel", ct.c_uint32),
        ("sat_id", ct.c_int32),
        ("freq", ct.c_double),
        ("prediction", ct.c_double),
        ("usable", ct.c_bool),
    ]

    def __init__(self):
        self.timestamp_week = 0
        self.timestamp_second = 0.0
        self.constel = constants.E_CONSTELLATION.E_CONSTEL_NONE
        self.sat_id = -1
        self.freq = 0.0
        self.prediction = 0.0
        self.usable = False


PE_API_FeaturesAI = (PE_FeaturesMultipathAI * constants.NUM_CHANNELS) * (
    constants.MAX_SATS
)
PE_API_PredictionsAI = (PE_PredictionsAI * constants.NUM_CHANNELS) * (
    constants.MAX_SATS
)


def ResetStruct(
    struct: Union[UsedSatellites, ProtectionLevel_pos, PE_Output_str, PE_API_FeaturesAI]  # type: ignore
):
    if isinstance(struct, UsedSatellites):
        struct.status = False
        struct.PRN = 0
        struct.elevation = 0.0
        struct.azimuth = 0.0
        struct.SNR_1 = 0.0
        struct.SNR_2 = 0.0
        struct.CodeResidual_1 = 0.0
        struct.CodeResidual_2 = 0.0
        struct.signal_id_1 = Signal_Obs.SIG_UNKNOWN
        struct.signal_id_2 = Signal_Obs.SIG_UNKNOWN

    elif isinstance(struct, ProtectionLevel_pos):
        struct.along = 0.0
        struct.cross = 0.0
        struct.vertical = 0.0
        struct.heading = 0.0
        struct.horizontal_flag = False
        struct.vertical_flag = False
        struct.heading_flag = False

    elif isinstance(struct, PE_Output_str):
        struct.timestamp_week = 0
        struct.timestamp_second = 0.0
        struct.GM_time = 0.0

        struct.pe_pos_out.pos_out = (ct.c_double * 3)()
        struct.pe_pos_out.latitude = 0.0
        struct.pe_pos_out.longitude = 0.0
        struct.pe_pos_out.height = 0.0
        struct.pe_pos_out.altitude = 0.0
        struct.pe_pos_out.latitude_direction = Latitude_Direction.NORTH
        struct.pe_pos_out.longitude_direction = Longitude_Direction.EAST
        struct.pe_pos_out.latitude_sigma = 0.0
        struct.pe_pos_out.longitude_sigma = 0.0
        struct.pe_pos_out.horizontal_sigma = 0.0
        struct.pe_pos_out.height_sigma = 0.0
        struct.pe_pos_out.altitude_sigma = 0.0
        struct.pe_pos_out.heading = 0.0
        struct.pe_pos_out.heading_flag = False
        struct.pe_pos_out.heading_sigma = 0.0
        struct.pe_pos_out.along_track_sigma = 0.0
        struct.pe_pos_out.cross_track_sigma = 0.0

        struct.pe_vel_out.vel_out = (ct.c_double * 3)()
        struct.pe_vel_out.velocityHorizontal = 0.0
        struct.pe_vel_out.velocityVertical = 0.0
        struct.pe_vel_out.TrackAngle = 0.0
        struct.pe_vel_out.velocityModule_knots = 0.0
        struct.pe_vel_out.velocityModule_kmh = 0.0
        struct.pe_vel_out.horizontal_sigma = 0.0
        struct.pe_vel_out.vertical_sigma = 0.0
        struct.pe_vel_out.track_angle_sigma = 0.0

        struct.pe_receiver_clock_out.ClockBias = 0.0
        struct.pe_receiver_clock_out.ClockDrift = 0.0

        ResetStruct(struct.pe_integrity_pos.TIR_10_3)
        ResetStruct(struct.pe_integrity_pos.TIR_10_4)
        ResetStruct(struct.pe_integrity_pos.TIR_10_5)
        ResetStruct(struct.pe_integrity_pos.TIR_10_6)
        ResetStruct(struct.pe_integrity_pos.TIR_0_003)

        struct.pe_integrity_vel.PL_horizontal_velocity = 0.0
        struct.pe_integrity_vel.PL_vertical_velocity = 0.0
        struct.pe_integrity_vel.PL_courseangle = 0.0
        struct.pe_integrity_vel.PL_horizontal_velocity_flag = False
        struct.pe_integrity_vel.PL_vertical_velocity_flag = False
        struct.pe_integrity_vel.PL_courseangle_flag = False

        struct.pe_solution_info.systemStatus = SystemStatus.NA_SYSTEM_STATUS
        struct.pe_solution_info.SSM_Signal = SafeStateMachineSignal.NO_SOLUTION
        struct.pe_solution_info.sensorStatus = SensorStatus.SYSTEM_OK
        struct.pe_solution_info.nominalState = NominalState.STATE_255_NO_SOLUTION
        struct.pe_solution_info.PDOP = 0.0
        struct.pe_solution_info.HDOP = 0.0
        struct.pe_solution_info.VDOP = 0.0
        struct.pe_solution_info.total_number_SVS_visible = 0
        struct.pe_solution_info.SVG_visible = 0
        struct.pe_solution_info.SVE_visible = 0
        struct.pe_solution_info.total_used_sats_fixed = 0
        struct.pe_solution_info.total_used_sats_float = 0
        struct.pe_solution_info.total_used_sats_float_one_freq = 0
        struct.pe_solution_info.gnssReferencePoint = (
            gnssReferencePoint.L1E1_ANTENNA_PHASE_CENTER
        )
        struct.pe_solution_info.antennaAxleOffsetX = 0.0
        struct.pe_solution_info.antennaAxleOffsetY = 0.0
        struct.pe_solution_info.antennaAxleOffsetZ = 0.0
        struct.pe_solution_info.vel_core_validity = False
        struct.pe_solution_info.SSM_Signal_Vel = SafeStateMachineSignal.NO_SOLUTION
        struct.pe_solution_info.correctionAge = 0.0
        struct.pe_solution_info.timeWithoutSatelliteSignals = 0.0

        for i in range(len(struct.pe_solution_info.used_sats_info_gps)):
            ResetStruct(struct.pe_solution_info.used_sats_info_gps[i])

        for i in range(len(struct.pe_solution_info.used_sats_info_gal)):
            ResetStruct(struct.pe_solution_info.used_sats_info_gal[i])

        for i in range(len(struct.pe_solution_info.used_sats_info_bds)):
            ResetStruct(struct.pe_solution_info.used_sats_info_bds[i])

        struct.pe_imu_odo_out.acc_bias = (ct.c_double * PE_imu_odo_out_arrays_size)()
        struct.pe_imu_odo_out.gyr_bias = (ct.c_double * PE_imu_odo_out_arrays_size)()
        struct.pe_imu_odo_out.att = (ct.c_double * PE_imu_odo_out_arrays_size)()
        struct.pe_imu_odo_out.scale_factor = 0.0

    elif isinstance(struct, PE_API_FeaturesAI):
        for sat_idx in range(len(struct)):
            for signal_idx in range(constants.NUM_CHANNELS):
                struct[sat_idx][signal_idx].usable = False
                struct[sat_idx][signal_idx].timestamp_week = 0
                struct[sat_idx][signal_idx].timestamp_second = 0
                struct[sat_idx][signal_idx].sat_id = -1
                struct[sat_idx][
                    signal_idx
                ].constel = constants.E_CONSTELLATION.E_CONSTEL_NONE
                struct[sat_idx][signal_idx].freq = 0.0
                struct[sat_idx][signal_idx].code = 0.0
                struct[sat_idx][signal_idx].phase = 0.0
                struct[sat_idx][signal_idx].doppler = 0.0
                struct[sat_idx][signal_idx].snr = 0.0
                struct[sat_idx][signal_idx].elevation = 0.0
                struct[sat_idx][signal_idx].residual = 0.0
                struct[sat_idx][signal_idx].iono = 0.0
