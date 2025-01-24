import ctypes as ct
import os
from configparser import ConfigParser
from typing import Tuple

from navutils.logger import Logger
from navutils.singleton import Singleton
from pewrapper.api.common_api_types import (
    Log_Handle,
    Signal_Obs,
    f_handle_bin_ephem_NVM,
)
from pewrapper.api.pe_api_types import (
    Configuration_info,
    GalCorrectionDataType,
    GNSSProtocol,
    doppler_sign,
    f_handle_ReportDTCStatus,
    type_receiver,
)
from pewrapper.handles import PE_LogWrapper, parse_binEphem
from pewrapper.managers.dtc_mgr import DtcManager
from pewrapper.misc.parser_utils import (
    ConfigDefaultValue,
    ParseBoolConfigurationField,
    ParseConfiguration,
    ParseConfigurationField,
)
from pewrapper.misc.utils import deepcopy_config, get_pe_api_category
from pewrapper.types.constants import SECURITY
from pewrapper.types.gps_time_wrapper import GPS_Time


class ConfigurationManager(metaclass=Singleton):

    def __init__(
        self,
        log_path: str,
        tracing_config_file: str,
        log_handle: Log_Handle = PE_LogWrapper,  # type: ignore
        binEphem_handle: f_handle_bin_ephem_NVM = parse_binEphem,  # type: ignore
        ReportDTCStatus: f_handle_ReportDTCStatus = DtcManager.ReportDTCStatus,  # type: ignore
    ):
        self.config_info_ = Configuration_info(
            binEphem_handle=binEphem_handle,
            log_handle=log_handle,
            ReportDTCStatus=ReportDTCStatus,
        )
        self.config_info_.log_path = log_path.encode("utf-8")
        self.config_info_.tracing_config_file = tracing_config_file.encode("utf-8")
        self.config_info_.log_category = get_pe_api_category(Logger.get_category())

        self.compute_log_ = False
        self.reconvergences_ = False
        self.reconvergences_multiple_files_ = False
        self.reconvergences_reset_rate_ = 0.0
        self.reconvergence_last_epoch_ = GPS_Time()
        self.IMU_latency_ = 0.0
        self.ODO_latency_ = 0.0
        self.use_IMU_qualifier_ = False

    def __getstate__(self):
        state = self.__dict__.copy()

        config_struct = state["config_info_"]

        config_info = {
            "parallelize": config_struct.parallelize,
            "max_num_pe_log_files": config_struct.max_num_pe_log_files,
            "max_num_wrapper_data_files": state[
                "config_info_"
            ].max_num_wrapper_data_files,
            "log_path": config_struct.log_path.decode("utf-8"),
            "tracing_config_file": config_struct.tracing_config_file.decode("utf-8"),
            "algo_profile": config_struct.algo_profile,
            "doppler_sign": config_struct.doppler_sign,
            "use_velCore": config_struct.use_velCore,
            "require_e2e_msg": config_struct.require_e2e_msg,
            "require_timesync": config_struct.require_timesync,
            "use_global_iono": config_struct.use_global_iono,
            "use_iono": config_struct.use_iono,
            "cn0_threshold_signal_1": config_struct.cn0_threshold_signal_1,
            "cn0_threshold_signal_2": config_struct.cn0_threshold_signal_2,
            "Signal_1_GAL": config_struct.Signal_1_GAL,
            "Signal_2_GAL": config_struct.Signal_2_GAL,
            "Signal_1_GPS": config_struct.Signal_1_GPS,
            "Signal_2_GPS": config_struct.Signal_2_GPS,
            "Signal_1_BDS": config_struct.Signal_1_BDS,
            "Signal_2_BDS": config_struct.Signal_2_BDS,
            "gnssProtocol": config_struct.gnssProtocol,
            "galCorrectionDataType": config_struct.galCorrectionDataType,
            "use_gps": config_struct.use_gps,
            "use_gal": config_struct.use_gal,
            "use_bds": config_struct.use_bds,
            "use_AI_multipath": config_struct.use_AI_multipath,
            "use_imu": config_struct.use_imu,
            "use_wt": config_struct.use_wt,
            "use_Integrity": config_struct.use_Integrity,
            "require_CS_integrity": config_struct.require_CS_integrity,
            "use_qm_variant": config_struct.use_qm_variant,
            "perform_js_checks": config_struct.perform_js_checks,
            "imu_2_antenna_lever_arm": list(config_struct.imu_2_antenna_lever_arm),
            "antenna_2_rear_axle": list(config_struct.antenna_2_rear_axle),
            "RearWheelDistance": config_struct.RearWheelDistance,
            "WheelDiameter": config_struct.WheelDiameter,
            "tile_hysteresis_delta_km": config_struct.tile_hysteresis_delta_km,
            "doppler_available": config_struct.doppler_available,
            "max_age_of_corrections": config_struct.max_age_of_corrections,
            "binEphem_handle": None,
            "log_handle": None,
            "log_category": config_struct.log_category,
            "ReportDTCStatus": None,
        }
        if SECURITY:
            config_info["license_file_path"] = config_struct.license_file_path.decode(
                "utf-8"
            )

        state["config_info_"] = config_info

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        serialized_config_info = self.__dict__["config_info_"]

        config_info = Configuration_info(
            binEphem_handle=parse_binEphem,
            log_handle=PE_LogWrapper,
            ReportDTCStatus=DtcManager.ReportDTCStatus,
        )

        config_info.parallelize = serialized_config_info["parallelize"]
        config_info.max_num_pe_log_files = serialized_config_info[
            "max_num_pe_log_files"
        ]
        config_info.max_num_wrapper_data_files = serialized_config_info[
            "max_num_wrapper_data_files"
        ]
        config_info.log_path = serialized_config_info["log_path"].encode("utf-8")
        config_info.tracing_config_file = serialized_config_info[
            "tracing_config_file"
        ].encode("utf-8")
        config_info.algo_profile = serialized_config_info["algo_profile"]
        config_info.doppler_sign = serialized_config_info["doppler_sign"]
        config_info.use_velCore = serialized_config_info["use_velCore"]
        config_info.require_e2e_msg = serialized_config_info["require_e2e_msg"]
        config_info.require_timesync = serialized_config_info["require_timesync"]
        config_info.use_global_iono = serialized_config_info["use_global_iono"]
        config_info.use_iono = serialized_config_info["use_iono"]
        config_info.cn0_threshold_signal_1 = serialized_config_info[
            "cn0_threshold_signal_1"
        ]
        config_info.cn0_threshold_signal_2 = serialized_config_info[
            "cn0_threshold_signal_2"
        ]
        config_info.Signal_1_GAL = serialized_config_info["Signal_1_GAL"]
        config_info.Signal_2_GAL = serialized_config_info["Signal_2_GAL"]
        config_info.Signal_1_GPS = serialized_config_info["Signal_1_GPS"]
        config_info.Signal_2_GPS = serialized_config_info["Signal_2_GPS"]
        config_info.Signal_1_BDS = serialized_config_info["Signal_1_BDS"]
        config_info.Signal_2_BDS = serialized_config_info["Signal_2_BDS"]
        config_info.gnssProtocol = serialized_config_info["gnssProtocol"]
        config_info.galCorrectionDataType = serialized_config_info[
            "galCorrectionDataType"
        ]
        config_info.use_gps = serialized_config_info["use_gps"]
        config_info.use_gal = serialized_config_info["use_gal"]
        config_info.use_bds = serialized_config_info["use_bds"]
        config_info.use_AI_multipath = serialized_config_info["use_AI_multipath"]
        config_info.use_imu = serialized_config_info["use_imu"]
        config_info.use_wt = serialized_config_info["use_wt"]
        config_info.use_Integrity = serialized_config_info["use_Integrity"]
        config_info.require_CS_integrity = serialized_config_info[
            "require_CS_integrity"
        ]
        config_info.use_qm_variant = serialized_config_info["use_qm_variant"]
        config_info.perform_js_checks = serialized_config_info["perform_js_checks"]

        imu_list = serialized_config_info["imu_2_antenna_lever_arm"]
        antenna_list = serialized_config_info["antenna_2_rear_axle"]
        config_info.imu_2_antenna_lever_arm = (ct.c_double * 3)(*imu_list)
        config_info.antenna_2_rear_axle = (ct.c_double * 3)(*antenna_list)

        config_info.RearWheelDistance = serialized_config_info["RearWheelDistance"]
        config_info.WheelDiameter = serialized_config_info["WheelDiameter"]
        config_info.tile_hysteresis_delta_km = serialized_config_info[
            "tile_hysteresis_delta_km"
        ]
        config_info.doppler_available = serialized_config_info["doppler_available"]
        config_info.max_age_of_corrections = serialized_config_info[
            "max_age_of_corrections"
        ]
        config_info.log_category = serialized_config_info["log_category"]

        if SECURITY and "license_file_path" in serialized_config_info:
            config_info.license_file_path = serialized_config_info[
                "license_file_path"
            ].encode("utf-8")

        self.config_info_ = config_info

    def reset(
        self,
        log_path: str,
        tracing_config_file: str,
        log_handle: Log_Handle = None,  # type: ignore
        binEphem_handle: f_handle_bin_ephem_NVM = None,  # type: ignore
        ReportDTCStatus: f_handle_ReportDTCStatus = None,  # type: ignore
    ):
        self.__init__(
            log_path,
            tracing_config_file,
            log_handle=(
                self.config_info_.log_handle if log_handle is None else log_handle
            ),
            binEphem_handle=(
                self.config_info_.binEphem_handle
                if binEphem_handle is None
                else binEphem_handle
            ),
            ReportDTCStatus=(
                self.config_info_.ReportDTCStatus
                if ReportDTCStatus is None
                else ReportDTCStatus
            ),
        )

    def reset_log_path(self, log_path: str):
        self.config_info_.log_path = log_path.encode("utf-8")

    def parse_config_file(self, filename: str, verbose=True) -> Tuple[bool, str]:
        addInfo = ""

        if verbose:
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.MAIN,
                f" Parsing wrapper config PE_Wrapper: {filename}",
            )

        if not os.path.isfile(filename):
            addInfo = "Config PE_Wrapper file doesn't exist."
            return False, addInfo

        pt = ConfigParser()
        pt.read(filename)

        returnValue = True

        # if SECURITY:
        #     # LICENSE PATH
        #     success, addInfo, self.config_info_.license_file_path = ParseConfiguration(
        #         self.config_info_.license_file_path,
        #         pt,
        #         "Configuration.license_file_path",
        #         "license_file_path",
        #         addInfo,
        #         verbose=verbose,
        #     )

        # CHANNEL CONFIGURATION
        success, addInfo, self.config_info_.Signal_1_GAL = ParseConfiguration(
            self.config_info_.Signal_1_GAL,
            pt,
            "Configuration.gal_channel_1",
            "GAL Channel 1 signal",
            addInfo,
            True,
            Signal_Obs.SIG_UNKNOWN,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.Signal_2_GAL = ParseConfiguration(
            self.config_info_.Signal_2_GAL,
            pt,
            "Configuration.gal_channel_2",
            "GAL Channel 2 signal",
            addInfo,
            True,
            Signal_Obs.SIG_UNKNOWN,
            True,
            Signal_Obs.GAL_E5B_Q,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.Signal_1_GPS = ParseConfiguration(
            self.config_info_.Signal_1_GPS,
            pt,
            "Configuration.gps_channel_1",
            "GPS Channel 1 signal",
            addInfo,
            True,
            Signal_Obs.SIG_UNKNOWN,
            True,
            Signal_Obs.GPS_L1_C,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.Signal_2_GPS = ParseConfiguration(
            self.config_info_.Signal_2_GPS,
            pt,
            "Configuration.gps_channel_2",
            "GPS Channel 2 signal",
            addInfo,
            True,
            Signal_Obs.SIG_UNKNOWN,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.Signal_1_BDS = ParseConfiguration(
            self.config_info_.Signal_1_BDS,
            pt,
            "Configuration.bds_channel_1",
            "BDS Channel 1 signal",
            addInfo,
            True,
            Signal_Obs.SIG_UNKNOWN,
            True,
            Signal_Obs.BDS_B1I_D1,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.Signal_2_BDS = ParseConfiguration(
            self.config_info_.Signal_2_BDS,
            pt,
            "Configuration.bds_channel_2",
            "BDS Channel 2 signal",
            addInfo,
            True,
            Signal_Obs.SIG_UNKNOWN,
            True,
            Signal_Obs.BDS_B2A_P,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # GNSS PROTOCOL
        success, addInfo, self.config_info_.gnssProtocol = ParseConfiguration(
            self.config_info_.gnssProtocol,
            pt,
            "Configuration.gnss_protocol",
            "GNSS protocol",
            addInfo,
            True,
            GNSSProtocol.PROTOCOL_UNKNOWN,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # GAL CORRECTION DATA TYPE
        success, addInfo, self.config_info_.galCorrectionDataType = ParseConfiguration(
            self.config_info_.galCorrectionDataType,
            pt,
            "Configuration.gal_correction_type",
            "GAL correction data type",
            addInfo,
            True,
            GalCorrectionDataType.CORRECTION_DATA_UNKNOWN,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # CONSTEL CONFIGURATION
        success, addInfo, self.config_info_.use_gps = ParseBoolConfigurationField(
            self.config_info_.use_gps,
            pt,
            "Configuration.use_gps",
            "Use GPS",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.use_gal = ParseBoolConfigurationField(
            self.config_info_.use_gal,
            pt,
            "Configuration.use_gal",
            "Use GAL",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.use_bds = ParseBoolConfigurationField(
            self.config_info_.use_bds,
            pt,
            "Configuration.use_bds",
            "Use BDS",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # RECEIVER CONFIGURATION
        success, addInfo, self.config_info_.algo_profile = ParseConfiguration(
            self.config_info_.algo_profile,
            pt,
            "Configuration.algo_profile",
            "Algo profile",
            addInfo,
            True,
            type_receiver.UNKNOWN,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # DOPPLER CONFIGURATION
        success, addInfo, self.config_info_.doppler_sign = ParseConfiguration(
            self.config_info_.doppler_sign,
            pt,
            "Configuration.doppler_sign",
            "Doppler sign",
            addInfo,
            True,
            doppler_sign.UNKNOWN_SIGN,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # MAX AGE OF CORRECTIONS
        (
            success,
            addInfo,
            self.config_info_.max_age_of_corrections,
        ) = ParseConfigurationField(
            self.config_info_.max_age_of_corrections,
            pt,
            "Configuration.max_age_of_corrections",
            "Max Age of Corrections",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # FEATURES CONFIGURATION
        (
            success,
            addInfo,
            self.config_info_.use_AI_multipath,
        ) = ParseBoolConfigurationField(
            self.config_info_.use_AI_multipath,
            pt,
            "Configuration.use_AI_multipath",
            "Use AI Multipath",
            addInfo,
            ConfigDefaultValue.default_value_false,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.use_imu = ParseBoolConfigurationField(
            self.config_info_.use_imu,
            pt,
            "Configuration.use_imu",
            "Use IMU",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.use_wt = ParseBoolConfigurationField(
            self.config_info_.use_wt,
            pt,
            "Configuration.use_wt",
            "Use WheelTicks",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.use_qm_variant,
        ) = ParseBoolConfigurationField(
            self.config_info_.use_qm_variant,
            pt,
            "Configuration.use_qm_variant",
            "Use QM Variant",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.doppler_available,
        ) = ParseBoolConfigurationField(
            self.config_info_.doppler_available,
            pt,
            "Configuration.doppler_available",
            "Doppler available",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.perform_js_checks,
        ) = ParseBoolConfigurationField(
            self.config_info_.perform_js_checks,
            pt,
            "Configuration.perform_js_checks",
            "Perform Jamming-Spoofing checks",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.use_velCore = ParseBoolConfigurationField(
            self.config_info_.use_velCore,
            pt,
            "Configuration.use_velCore",
            "Use VelCore",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        success, addInfo, self.config_info_.use_Integrity = ParseBoolConfigurationField(
            self.config_info_.use_Integrity,
            pt,
            "Configuration.use_Integrity",
            "Use Integrity",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # MESSAGES DATA CONFIGURATION
        (
            success,
            addInfo,
            self.config_info_.require_e2e_msg,
        ) = ParseBoolConfigurationField(
            self.config_info_.require_e2e_msg,
            pt,
            "Configuration.require_e2e_msg",
            "Require E2E msg",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        if (
            self.config_info_.require_e2e_msg
            or self.config_info_.algo_profile == type_receiver.E_UBLOX_A9
        ):
            (
                success,
                addInfo,
                self.config_info_.require_timesync,
            ) = ParseBoolConfigurationField(
                self.config_info_.require_timesync,
                pt,
                "Configuration.require_timesync",
                "Require timesync",
                addInfo,
                verbose=verbose,
            )
            returnValue = returnValue and success
            if not returnValue:
                return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.use_global_iono,
        ) = ParseBoolConfigurationField(
            self.config_info_.use_global_iono,
            pt,
            "Configuration.use_global_iono",
            "Use Global Iono instead of Regional",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.use_iono,
        ) = ParseBoolConfigurationField(
            self.config_info_.use_iono,
            pt,
            "Configuration.use_iono",
            "Use Iono",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # CN0 CONFIGURATION
        (
            success,
            addInfo,
            self.config_info_.cn0_threshold_signal_1,
        ) = ParseConfigurationField(
            self.config_info_.cn0_threshold_signal_1,
            pt,
            "Configuration.cn0_threshold_signal_1",
            "Cn0 Threshold Signal 1",
            addInfo,
            True,
            32.0,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.cn0_threshold_signal_2,
        ) = ParseConfigurationField(
            self.config_info_.cn0_threshold_signal_2,
            pt,
            "Configuration.cn0_threshold_signal_2",
            "Cn0 Threshold Signal 2",
            addInfo,
            True,
            32.0,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.require_CS_integrity,
        ) = ParseBoolConfigurationField(
            self.config_info_.require_CS_integrity,
            pt,
            "Configuration.require_CS_integrity",
            "Require CS integrity messages",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # CAR DIMENSIONS CONFIGURATION
        (
            success,
            addInfo,
            self.config_info_.imu_2_antenna_lever_arm,
        ) = ParseConfigurationField(
            self.config_info_.imu_2_antenna_lever_arm,
            pt,
            "Configuration.imu_2_antenna_lever_arm",
            "Lever Arm",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.antenna_2_rear_axle,
        ) = ParseConfigurationField(
            self.config_info_.antenna_2_rear_axle,
            pt,
            "Configuration.antenna_2_rear_axle",
            "Antenna position to rear axle",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.WheelDiameter,
        ) = ParseConfigurationField(
            self.config_info_.WheelDiameter,
            pt,
            "Configuration.WheelDiameter",
            "Wheel Diameter",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.RearWheelDistance,
        ) = ParseConfigurationField(
            self.config_info_.RearWheelDistance,
            pt,
            "Configuration.RearWheelDistance",
            "Rear axis to wheel distance",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        # OTHER
        (
            success,
            addInfo,
            self.config_info_.tile_hysteresis_delta_km,
        ) = ParseConfigurationField(
            self.config_info_.tile_hysteresis_delta_km,
            pt,
            "Configuration.tile_hysteresis_delta_km",
            "Tile hysteresis",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.config_info_.max_num_pe_log_files,
        ) = ParseConfigurationField(
            self.config_info_.max_num_pe_log_files,
            pt,
            "Output_Files_Config.max_num_pe_log_files",
            "Max number of pe_log files",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        self.config_info_.parallelize = False

        # WRAPPER CONFIGURATION
        (
            success,
            addInfo,
            self.use_IMU_qualifier_,
        ) = ParseBoolConfigurationField(
            self.use_IMU_qualifier_,
            pt,
            "Configuration.use_IMU_qualifier",
            "Use IMU qualifier",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.IMU_latency_,
        ) = ParseConfigurationField(
            self.IMU_latency_,
            pt,
            "Configuration.IMU_latency",
            "IMU latency",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.ODO_latency_,
        ) = ParseConfigurationField(
            self.ODO_latency_,
            pt,
            "Configuration.ODO_latency",
            "ODO latency",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.compute_log_,
        ) = ParseBoolConfigurationField(
            self.compute_log_,
            pt,
            "Configuration.compute_log",
            "Compute log",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.reconvergences_,
        ) = ParseBoolConfigurationField(
            self.reconvergences_,
            pt,
            "Configuration.reconvergences",
            "Compute reconvergences",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.reconvergences_multiple_files_,
        ) = ParseBoolConfigurationField(
            self.reconvergences_multiple_files_,
            pt,
            "Configuration.reconvergences_multiple_files",
            "Multiple reconvergences files",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        (
            success,
            addInfo,
            self.reconvergences_reset_rate_,
        ) = ParseConfigurationField(
            self.reconvergences_reset_rate_,
            pt,
            "Configuration.reconvergences_reset_rate",
            "Reconvergences reset rate",
            addInfo,
            verbose=verbose,
        )
        returnValue = returnValue and success
        if not returnValue:
            return returnValue, addInfo

        if (
            (
                not self.compute_log_
                and self.config_info_.algo_profile != type_receiver.E_UBLOX_A9
            )
            and self.config_info_.require_timesync
            and self.config_info_.gnssProtocol == GNSSProtocol.UBX
        ):
            addInfo = "Require timesync is set to true and gnss protocol is set to UBX, but compute log is set to false in PE Config"
            return False, addInfo

        return returnValue, addInfo

    def get_config(
        self,
        base_path: str = None,
        use_ai_multipath: bool = None,
        generation: int = None,
    ) -> Configuration_info:
        config_copy = deepcopy_config(self.config_info_)
        if use_ai_multipath is not None:
            config_copy.use_AI_multipath = use_ai_multipath

            if use_ai_multipath:
                path_suffix = "AI"

                if generation is not None:
                    path_suffix = f"AI_generation{generation}"

                config_copy.log_path = os.path.join(
                    (
                        config_copy.log_path.decode("utf-8")
                        if base_path is None
                        else base_path
                    ),
                    path_suffix,
                ).encode("utf-8")

            else:
                config_copy.log_path = os.path.join(
                    config_copy.log_path.decode("utf-8"), "noAI"
                ).encode("utf-8")
        return config_copy
