import io
import os
from typing import List

from navutils.logger import Logger
from pewrapper.api import (
    Latitude_Direction,
    Longitude_Direction,
    NominalState,
    SafeState,
    UsedSatellites,
)
from pewrapper.managers import OutputStr
from pewrapper.misc import convert_deg_to_int_min_sec
from pewrapper.types import (
    COMPUTE_INPUT_DATA_WITH_TIME,
    COMPUTE_INPUT_DATA_WITH_TIME_DELAY_TAG,
    COMPUTE_INPUT_DATA_WITHOUT_TIME,
    SPEED_OF_LIGHT,
    GPS_Time,
)


class Position_Recorder:
    def __init__(self):
        self.__output_file = None
        self.output_file_name_ = "rtppp_"
        self.output_file_name_rearaxis_ = "rtppp_rearaxis_"
        self.output_file_path_ = ""

    def initialize(self, output_path: str, epoch_file: GPS_Time):
        self.output_file_path_ = os.path.join(
            output_path,
            self.output_file_name_ + self.get_date_filename(epoch_file) + ".txt",
        )
        try:
            self.__output_file = open(self.output_file_path_, "w")
        except IOError as e:
            Logger.log_message(
                Logger.Category.ERROR, Logger.Module.WRITER, f"Error opening file: {e}"
            )
            return False

        s_log = f" {self.output_file_path_} file opened"
        Logger.log_message(Logger.Category.INFO, Logger.Module.WRITER, s_log)

        return True

    def close_file(self):
        if self.__output_file is not None and not self.__output_file.closed:
            self.__output_file.close()

        return True

    @staticmethod
    def get_number_sats(table: List[UsedSatellites], attribute: str):
        return sum(1 for sat in table if getattr(sat, attribute))

    def write_pos_header(self, ai_pe_wrapper_commit: str, common_lib_commit: str):
        header_lines = [
            f"#Commit ID PE_Wrapper_develop: {ai_pe_wrapper_commit}\n",
            f"#Commit ID COMMON_LIB: {common_lib_commit}\n",
            f"#{'YEAR':^4}  {'MTH':^4}  {'DAY':^4}  {'HOU':^4}  {'MIN':^4}  {'SEC':^9}  ",
            f"{'X':^15}  {'Y':^15}  {'Z':^15}  {'LAT':^15}  {'LON':^15}  {'HEI':^8}  ",
            f"{'SIGLAT':^14}  {'SIGLON':^14}  {'SIGHEI':^14}  {'CLOCK':^16}  {'TROPO':^5}  ",
            f"{'ALL_Fix':^7}  {'SVG_Fix':^7}  {'SVE_Fix':^7}  {'SVC_Fix':^7}  {'ALL_Float':^9}  {'SVG_Float':^9}  {'SVE_Float':^9}  {'SVC_Float':^9}  {'ALL_One_Freq':^12}  {'SVG_One_Freq':^12}  {'SVE_One_Freq':^12}  {'SVC_One_Freq':^12}  ",
            f"{'DEGRADED':^8}  {'QUALITY':^7}  ",
            f"{'Vx':^15}  {'Vy':^15}  {'Vz':^15}  {'SIGVx':^5}  {'SIGVy':^5}  {'SIGVz':^5}  ",
            f"{'BiasAccX':^12}  {'BiasAccY':^12}  {'BiasAccZ':^12}  {'BiasGyrX':^12}  {'BiasGyrY':^12}  {'BiasGyrZ':^12}  {'Roll':^12}  {'Pitch':^12}  {'Heading':^12}  ",
            f"{'SF':^12}  {'HDOP':^12}  {'Altitude':^12}  {'TrackAngle':^12}  ",
            f"{'SAFE_STATE':^20}  {'NOMINAL_STATE':^13}  {'VALID_PL':^8}  ",
            f"{'PL_H (1e-6)':^14}  {'PL_V (1e-6)':^14}  {'PL_AT (1e-6)':^14}  {'PL_CT (1e-6)':^14}  {'PL_HEADING(1e-6)':^20}  ",
            f"{'PL_H (1e-5)':^14}  {'PL_V (1e-5)':^14}  {'PL_AT (1e-5)':^14}  {'PL_CT (1e-5)':^14}  {'PL_HEADING(1e-5)':^20}  ",
            f"{'PL_H (1e-4)':^14}  {'PL_V (1e-4)':^14}  {'PL_AT (1e-4)':^14}  {'PL_CT (1e-4)':^14}  {'PL_HEADING(1e-4)':^20}  ",
            f"{'PL_H (1e-3)':^14}  {'PL_V (1e-3)':^14}  {'PL_AT (1e-3)':^14}  {'PL_CT (1e-3)':^14}  {'PL_HEADING(1e-3)':^20}  ",
            f"{'PL_H (0.003)':^14}  {'PL_V (0.003)':^14}  {'PL_AT (0.003)':^14}  {'PL_CT (0.003)':^14}  {'PL_HEADING(0.003)':^20}  ",
            f"{'PL_H (0.32)':^14}  {'PL_V (0.32)':^14}  {'PL_AT (0.32)':^14}  {'PL_CT (0.32)':^14}  {'PL_HEADING(0.32)':^20}  ",
            f"{'PL_HV':^14}  {'PL_VV':^14}  {'PL_TA':^14}  {'IF_HV':^14}  {'IF_VV':^14}  {'IF_TA':^14}  ",
            f"{'VQ':^14}  {'VALID_HEADING':^14}  {'VEL_CORE_VALIDITY':^14}  \n",
            f"#{'':^39}  ",
            f"{'m':^15}  {'m':^15}  {'m':^15}  {'ddd mm ss.sssss':^15}  {'ddd mm ss.sssss':^15}  {'m':^8}  ",
            f"{'m':^14}  {'m':^14}  {'m':^14}  {'ns':^16}  {'m':^5}  ",
            f"{'':^77}  ",
            f"{'m/s':^15}  {'m/s':^15}  {'m/s':^15}  {'':^19}  ",
            f"{'m/s2':^12}  {'m/s2':^12}  {'m/s2':^12}  {'rad/s':^12}  {'rad/s':^12}  {'rad/s':^12}  {'deg':^12}  {'deg':^12}  {'deg':^12}  ",
            f"{'-':^12}  {'-':^12}  {'m':^12}  {'deg':^12}  ",
            f"{'':^45}  ",
            f"{'m':^14}  {'m':^14}  {'m':^14}  {'m':^14}  {'deg':^20}  ",
            f"{'m':^14}  {'m':^14}  {'m':^14}  {'m':^14}  {'deg':^20}  ",
            f"{'m':^14}  {'m':^14}  {'m':^14}  {'m':^14}  {'deg':^20}  ",
            f"{'m':^14}  {'m':^14}  {'m':^14}  {'m':^14}  {'deg':^20}  ",
            f"{'m':^14}  {'m':^14}  {'m':^14}  {'m':^14}  {'deg':^20}  ",
            f"{'m':^14}  {'m':^14}  {'m':^14}  {'m':^14}  {'deg':^20}  ",
            f"{'m/s':^14}  {'m/s':^14}  {'deg':^14}  {'-':^14}  {'-':^14}  {'-':^14}  ",
            f"{'-':^14}  {'-':^14}  {'-':^14}  \n",
            f"#{''.center(4,'-')}  {''.center(4,'-')}  {''.center(4,'-')}  {''.center(4,'-')}  {''.center(4,'-')}  {''.center(9,'-')}  ",
            f"{''.center(15,'-')}  {''.center(15,'-')}  {''.center(15,'-')}  {''.center(15,'-')}  {''.center(15,'-')}  {''.center(8,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(16,'-')}  {''.center(5,'-')}  ",
            f"{''.center(7,'-')}  {''.center(7,'-')}  {''.center(7,'-')}  {''.center(7,'-')}  {''.center(9,'-')}  {''.center(9,'-')}  {''.center(9,'-')}  {''.center(9,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  ",
            f"{''.center(8,'-')}  {''.center(7,'-')}  ",
            f"{''.center(15,'-')}  {''.center(15,'-')}  {''.center(15,'-')}  {''.center(5,'-')}  {''.center(5,'-')}  {''.center(5,'-')}  ",
            f"{''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  ",
            f"{''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  {''.center(12,'-')}  ",
            f"{''.center(20,'-')}  {''.center(13,'-')}  {''.center(8,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(20,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(20,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(20,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(20,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(20,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(20,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  ",
            f"{''.center(14,'-')}  {''.center(14,'-')}  {''.center(14,'-')}  \n",
        ]
        self.__output_file.write("".join(header_lines))
        self.__output_file.flush()

        return True

    def write_pos_epoch(self, date: GPS_Time, is_qm: bool, data: OutputStr):
        data_epoch = GPS_Time(
            w=data.output_PE.timestamp_week, s=data.output_PE.timestamp_second
        )

        if data_epoch == GPS_Time():
            return ""

        pe_pos_out = data.output_PE.pe_pos_out

        siLonDeg, siLonMin, dLonSec = convert_deg_to_int_min_sec(pe_pos_out.longitude)
        siLatDeg, siLatMin, dLatSec = convert_deg_to_int_min_sec(pe_pos_out.latitude)

        siLonDeg %= 360

        is_degraded = False

        s = io.StringIO()

        s.write(f"{data_epoch.calendar_column_str_d():^37}  ")
        s.write(
            f"{pe_pos_out.pos_out[0]:^15.4f}  {pe_pos_out.pos_out[1]:^15.4f}  {pe_pos_out.pos_out[2]:^15.4f}  "
        )
        if pe_pos_out.latitude_direction == Latitude_Direction.NORTH:
            s.write(f"{siLatDeg:^3} {siLatMin:^2} {dLatSec:^8.5f}  ")
        elif pe_pos_out.latitude_direction == Latitude_Direction.SOUTH:
            s.write(f"{-siLatDeg:^3} {siLatMin:^2} {dLatSec:^8.5f}  ")
        else:
            s.write(f"{0:^3} {0:^2} {0:^8.5f}  ")

        if pe_pos_out.longitude_direction == Longitude_Direction.EAST:
            s.write(f"{siLonDeg:^3} {siLonMin:^2} {dLonSec:^8.5f}  ")
        elif pe_pos_out.longitude_direction == Longitude_Direction.WEST:
            s.write(f"{-siLonDeg:^3} {siLonMin:^2} {dLonSec:^8.5f}  ")
        else:
            s.write(f"{0:^3} {0:^2} {0:^8.5f}  ")

        s.write(f"{pe_pos_out.height:^8.4f}  ")
        s.write(f"{pe_pos_out.latitude_sigma:^14.4f}  ")
        s.write(f"{pe_pos_out.longitude_sigma:^14.4f}  ")

        s.write(f"{pe_pos_out.height_sigma:^14.4f}  ")
        s.write(
            f"{(data.output_PE.pe_receiver_clock_out.ClockBias * 1e9 / SPEED_OF_LIGHT):^16.4f}  "
        )
        s.write(f"{0:^5}  ")

        # Total Fixed Satellites
        s.write(f"{data.output_PE.pe_solution_info.total_used_sats_fixed:^7}  ")
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_gps, 'fixed'):^7}  "
        )
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_gal, 'fixed'):^7}  "
        )
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_bds, 'fixed'):^7}  "
        )

        # Total Float Satellites
        s.write(f"{data.output_PE.pe_solution_info.total_used_sats_float:^9}  ")
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_gps, 'status'):^9}  "
        )
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_gal, 'status'):^9}  "
        )
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_bds, 'status'):^9}  "
        )

        # Total One Freq Satellites
        s.write(
            f"{data.output_PE.pe_solution_info.total_used_sats_float_one_freq:^12}  "
        )
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_gps, 'only_one_freq'):^12}  "
        )
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_gal, 'only_one_freq'):^12}  "
        )
        s.write(
            f"{self.get_number_sats(data.output_PE.pe_solution_info.used_sats_info_bds, 'only_one_freq'):^12}  "
        )

        s.write(f"{is_degraded:^7}  ")

        s.write(f"{data.output_PE.pe_solution_info.systemStatus:^8}  ")

        s.write(
            f"{data.output_PE.pe_vel_out.vel_out[0]:^15.4f}  {data.output_PE.pe_vel_out.vel_out[1]:^15.4f}  {data.output_PE.pe_vel_out.vel_out[2]:^15.4f}  "
        )
        s.write(f"{0:^5.4f}  {0:^5.4f}  {0:^5.4f}  ")

        s.write(
            f"{data.output_PE.pe_imu_odo_out.acc_bias[0]:^12.5f}  {data.output_PE.pe_imu_odo_out.acc_bias[1]:^12.5f}  {data.output_PE.pe_imu_odo_out.acc_bias[2]:^12.5f}  "
        )
        s.write(
            f"{data.output_PE.pe_imu_odo_out.gyr_bias[0]:^12.5f}  {data.output_PE.pe_imu_odo_out.gyr_bias[1]:^12.5f}  {data.output_PE.pe_imu_odo_out.gyr_bias[2]:^12.5f}  "
        )
        s.write(
            f"{data.output_PE.pe_imu_odo_out.att[0]:^12.5f}  {data.output_PE.pe_imu_odo_out.att[1]:^12.5f}  {data.output_PE.pe_imu_odo_out.att[2]:^12.5f}  "
        )

        s.write(f"{data.output_PE.pe_imu_odo_out.scale_factor:^12.5f}  ")

        s.write(f"{data.output_PE.pe_solution_info.HDOP:^12.4f}  ")
        s.write(f"{data.output_PE.pe_pos_out.altitude:^12.4f}  ")
        s.write(f"{data.output_PE.pe_vel_out.TrackAngle:^12.4f}  ")

        if (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_255_NO_SOLUTION
            and is_qm
        ):
            data.SafeState = SafeState.NO_SOLUTION_STATE

        if data.SafeState == SafeState.ERROR_STATE:
            pe_state = f"{'ERROR_STATE':^20}"
            vq_state = f"{'6':^14}"
        elif data.SafeState == SafeState.INACTIVE_STATE:
            pe_state = f"{'INACTIVE_STATE':^20}"
            vq_state = f"{'15':^14}"
        elif data.SafeState == SafeState.INIT_STATE:
            pe_state = f"{'INIT_STATE':^20}"
            vq_state = f"{'15':^14}"
        elif data.SafeState == SafeState.NO_SOLUTION_STATE:
            pe_state = f"{'NO_SOLUTION_STATE':^20}"
            vq_state = f"{'14':^14}"
        elif data.SafeState == SafeState.INIT_MON_STATE:
            pe_state = f"{'INIT_MON_STATE':^20}"
            vq_state = f"{'13':^14}"
        elif data.SafeState == SafeState.SAFE_SOLUTION_STATE:
            pe_state = f"{'SAFE_SOLUTION_STATE':^20}"
            vq_state = f"{'10':^14}"
        elif data.SafeState == SafeState.VALID_SOLUTION_STATE:
            pe_state = f"{'VALID_SOLUTION_STATE':^20}"
            vq_state = f"{'9':^14}"
        else:
            pe_state = f"{'':^20}"
            vq_state = f"{'':^14}"

        s.write(f"{pe_state}  ")

        if data.output_PE.pe_solution_info.nominalState == NominalState.STATE_0_INIT:
            nominal_state = f"{'0':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_1_LANE_PRECISE
        ):
            nominal_state = f"{'1':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_2_ROAD_PRECISE
        ):
            nominal_state = f"{'2':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_3_1_SHORT_DR
        ):
            nominal_state = f"{'3.1':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_3_2_LONG_DR
        ):
            nominal_state = f"{'3.2':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_3_3_UNLIMITED_DR
        ):
            nominal_state = f"{'3.3':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_4_1_DEGRADED_ACCURACY
        ):
            nominal_state = f"{'4.1':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_4_2_CS_OUTAGE
        ):
            nominal_state = f"{'4.2':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_4_3_EPHEM_ONLY
        ):
            nominal_state = f"{'4.3':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_4_4_DEGRADED_DR
        ):
            nominal_state = f"{'4.4':^13}"
        elif (
            data.output_PE.pe_solution_info.nominalState
            == NominalState.STATE_255_NO_SOLUTION
        ):
            nominal_state = f"{'255':^13}"
        else:
            nominal_state = f"{'':^13}"
        s.write(f"{nominal_state}  ")

        s.write(f"{0:^8}  ")

        if (
            data.output_PE.pe_integrity_pos.TIR_10_6.horizontal_flag
            and data.output_PE.pe_integrity_pos.TIR_10_6.vertical_flag
        ):
            s.write(f"{0.0:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_6.vertical:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_6.along:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_6.cross:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_6.heading:^20.4f}  ")
        else:
            s.write(f"{'':^84}  ")

        if (
            data.output_PE.pe_integrity_pos.TIR_10_5.horizontal_flag
            and data.output_PE.pe_integrity_pos.TIR_10_5.vertical_flag
        ):
            s.write(f"{0.0:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_5.vertical:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_5.along:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_5.cross:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_5.heading:^20.4f}  ")
        else:
            s.write(f"{'':^84}  ")

        if (
            data.output_PE.pe_integrity_pos.TIR_10_4.horizontal_flag
            and data.output_PE.pe_integrity_pos.TIR_10_4.vertical_flag
        ):
            s.write(f"{0.0:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_4.vertical:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_4.along:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_4.cross:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_4.heading:^20.4f}  ")
        else:
            s.write(f"{'':^84}  ")

        if (
            data.output_PE.pe_integrity_pos.TIR_10_3.horizontal_flag
            and data.output_PE.pe_integrity_pos.TIR_10_3.vertical_flag
        ):
            s.write(f"{0.0:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_3.vertical:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_3.along:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_3.cross:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_10_3.heading:^20.4f}  ")
        else:
            s.write(f"{'':^84}  ")

        if (
            data.output_PE.pe_integrity_pos.TIR_0_003.horizontal_flag
            and data.output_PE.pe_integrity_pos.TIR_0_003.vertical_flag
        ):
            s.write(f"{0.0:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_0_003.vertical:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_0_003.along:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_0_003.cross:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_pos.TIR_0_003.heading:^20.4f}  ")
        else:
            s.write(f"{'':^84}  ")

        if (
            data.output_PE.pe_integrity_pos.TIR_0_003.horizontal_flag
            and data.output_PE.pe_integrity_pos.TIR_0_003.vertical_flag
        ):
            s.write(f"{0.0:^14.4f}  ")
            s.write(f"{data.output_PE.pe_pos_out.height_sigma:^14.4f}  ")
            s.write(f"{data.output_PE.pe_pos_out.longitude_sigma:^14.4f}  ")
            s.write(f"{data.output_PE.pe_pos_out.latitude_sigma:^14.4f}  ")
            s.write(f"{data.output_PE.pe_pos_out.heading_sigma:^20.4f}  ")
        else:
            s.write(f"{'':^84}  ")

        if (
            data.output_PE.pe_integrity_pos.TIR_0_003.horizontal_flag
            and data.output_PE.pe_integrity_pos.TIR_0_003.vertical_flag
        ):
            s.write(
                f"{data.output_PE.pe_integrity_vel.PL_horizontal_velocity:^14.4f}  "
            )
            s.write(f"{data.output_PE.pe_integrity_vel.PL_vertical_velocity:^14.4f}  ")
            s.write(f"{data.output_PE.pe_integrity_vel.PL_courseangle:^14.4f}  ")
            s.write(
                f"{data.output_PE.pe_integrity_vel.PL_horizontal_velocity_flag:^14.4f}  "
            )
            s.write(
                f"{data.output_PE.pe_integrity_vel.PL_vertical_velocity_flag:^14.4f}  "
            )
            s.write(f"{data.output_PE.pe_integrity_vel.PL_courseangle_flag:^14.4f}  ")
        else:
            s.write(f"{'':^94}  ")

        s.write(f"{vq_state:^14}")

        if (
            data.output_PE.pe_solution_info.nominalState != NominalState.STATE_0_INIT
            and data.output_PE.pe_solution_info.nominalState
            != NominalState.STATE_255_NO_SOLUTION
            and data.output_PE.pe_pos_out.heading_sigma > 0
        ):
            heading_valid = f"{'1':^14}"
        else:
            heading_valid = f"{'1':^14}"
        s.write(f"{heading_valid:^14}")

        if data.output_PE.pe_solution_info.vel_core_validity:
            vel_core_validity_rtppp = f"{'1':^14}"
        else:
            vel_core_validity_rtppp = f"{'0':^14}"
        s.write(f"{vel_core_validity_rtppp:^14}")

        self.__output_file.write(f"{s.getvalue()}\n")
        self.__output_file.flush()

        return True

    @staticmethod
    def get_date_filename(date: GPS_Time):
        return f"{date.year()}{date.month()}{date.day()}_{date.hour()}{date.min()}{date.day_sec()}"
