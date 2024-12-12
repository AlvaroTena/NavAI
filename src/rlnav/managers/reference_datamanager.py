import pandas as pd

from navutils.logger import Logger
from navutils.singleton import Singleton
from rlnav.types.reference_types import SPAN_COLUMNS, ReferenceMode, ReferenceType


class ReferenceDataManager(metaclass=Singleton):
    def __init__(self, ref_mode, ref_type):
        if ref_mode not in [ReferenceMode.KINEMATIC, ReferenceMode.STATIC]:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.READER,
                f"Wrong ref_mode: {ref_mode}. The ref_mode options only accepts KINEMATIC and STATIC",
            )
            raise "Wrong ref_mode"

        if ref_type not in [
            ReferenceType.Kinematic.RTPPP_GSHARP,
            ReferenceType.Kinematic.RTPPP_PE,
            ReferenceType.Kinematic.RTPPP_MAGIC,
            ReferenceType.Kinematic.SPAN_FILE,
            ReferenceType.Kinematic.RTK_FILE,
            ReferenceType.Static.XYZ,
            ReferenceType.Static.LLH,
        ]:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.READER,
                f"Wrong ref_type: {ref_type}. The ref_type options only accepts RTPPP_GSHARP, RTPPP_PE, RTPPP_MAGIC, SPAN_FILE, RTK_FILE",
            )
            raise "Wrong ref_type"

        self.ref_mode = ref_mode
        self.ref_type = ref_type
        self.reference = pd.DataFrame()

    def reset(self, ref_mode, ref_type):
        self.__init__(ref_mode, ref_type)

    def parse_ref_data(self, filename):
        if self.ref_mode == ReferenceMode.KINEMATIC:
            if self.ref_type == ReferenceType.Kinematic.SPAN_FILE:
                self._parse_span_file(filename)

            elif self.ref_type in [
                ReferenceType.Kinematic.RTPPP_GSHARP,
                ReferenceType.Kinematic.RTPPP_PE,
                ReferenceType.Kinematic.RTPPP_MAGIC,
                ReferenceType.Kinematic.RTK_FILE,
            ]:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.READER,
                    f"ref_type: {self.ref_type}. Not implemented options RTPPP_GSHARP, RTPPP_PE, RTPPP_MAGIC, RTK_FILE",
                )
                return False

            else:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.READER,
                    f"Wrong ref_type in kinematic mode. The ref_type inputs accepted are: RTPPP_GSHARP, RTPPP_PE, RTPPP_MAGIC, SPAN_FILE, RTK_FILE",
                )
                return False

        elif self.ref_mode == ReferenceMode.STATIC:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.READER,
                f"ref_mode: {self.ref_mode}. Not implemented options STATIC",
            )
            return False

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.READER,
                f"Wrong ref_mode: {self.ref_mode}. The ref_mode options only accepts KINEMATIC and STATIC.",
            )
            return False

        return True

    def _parse_span_file(self, filename):
        if (
            self.ref_mode == ReferenceMode.KINEMATIC
            and self.ref_type == ReferenceType.Kinematic.SPAN_FILE
        ):
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.READER,
                f"Parsing SPAN file: {filename}",
            )

            df_span = pd.read_csv(
                filename,
                comment="#",
                skipfooter=54,
                sep="\s+",
                header=None,
                index_col=False,
                names=SPAN_COLUMNS,
                engine="python",
            )

            if df_span.empty:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.READER,
                    f"The SPAN_FILE ({filename}) is empty",
                )
                raise "SPAN_FILE bad formed"

            df_span["Epoch"] = pd.to_datetime(
                df_span["LocalDate"].astype(str) + " " + df_span["GPSTime"].astype(str),
                format="%Y/%m/%d %H:%M:%S.%f",
            )

            self.reference = df_span.drop(["LocalDate", "GPSTime"], axis=1)

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.READER,
                f"SPAN_FILE not compatible with {self.ref_mode} {self.ref_type}",
            )
