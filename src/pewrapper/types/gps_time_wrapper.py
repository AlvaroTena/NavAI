import copy
import math
import time
from datetime import datetime, timezone
from typing import Dict

from pewrapper.misc import getFloor, getInt

COMP_THRES = 1e-6

#  Number of days in each month.
#  @warning Month index starts in 1 (for January), so the month with 0 index,
#  has been asigned 0 days.
DAYS_IN_MONTH = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# Modified Julian date of year 2000
J2000 = 51544.5
# Offset in seconds between TT (Terrestrial Time) and TAI
TT_TAI_SECS = 32.184

# Conversion factor from MJD to JD
MJD_TO_JD = 2400000.5

# GST time zero epoch related to MJD (Modified Julian Date)
GST_TIME_START_MJD_DAYS = 44244

# Modified Julian date of year 1980
MJD_1980 = 44239
YEAR_1980 = 1980


class MJD_Time:
    # Seconds offset between GST time and TAI
    TAI_GPS_OFFSET = 19
    JULIAN_CENTURY = 36525.0

    SECONDS_IN_DAY = 86400
    DAYS_IN_WEEK = 7
    SECONDS_IN_WEEK = SECONDS_IN_DAY * DAYS_IN_WEEK
    GST_TIME_START_MJD_DAYS = 44244
    SECONDS_IN_HOUR = 3600
    SECONDS_IN_MINUTE = 60
    DAYS_IN_YEAR = 365

    day = 0
    second = 0.0

    def __init__(
        self,
        d=0,
        s=0.0,
        rhs=None,
        time=None,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
    ):
        def initialize_year_day_second(year, day, second):
            siMjdDayOffset = 0
            for siYearIndex in range(YEAR_1980, year):
                if not MJD_Time.isLeapYear(siYearIndex):
                    siNumDaysInYear = MJD_Time.DAYS_IN_YEAR
                else:
                    siNumDaysInYear = MJD_Time.DAYS_IN_YEAR + 1

                siMjdDayOffset += siNumDaysInYear

            siMjdDayOffset += day - 1

            self.day = int(MJD_1980 + siMjdDayOffset)
            self.second = float(second)

            self.normalize()

        if isinstance(rhs, MJD_Time):
            if self.__ne__(rhs):
                self.day = int(rhs.day)
                self.second = float(rhs.second)

        elif isinstance(time, GPS_Time):
            siWeekDays = time.second // MJD_Time.SECONDS_IN_DAY
            self.day = int(
                GST_TIME_START_MJD_DAYS + time.week * MJD_Time.DAYS_IN_WEEK + siWeekDays
            )
            self.second = float(time.second - siWeekDays * MJD_Time.SECONDS_IN_DAY)

            self.normalize()

        elif all(v is not None for v in [year, month, day, hour, minute, second]):
            siDayOfYear = 0
            bLeapYear = MJD_Time.isLeapYear(year)
            for siMonthIndex in range(1, month):
                siDayOfYear += DAYS_IN_MONTH[siMonthIndex]
                if siMonthIndex == 2 and bLeapYear:
                    siDayOfYear += 1
            siDayOfYear += day

            initialize_year_day_second(year, siDayOfYear, 0)

            daySeconds = (
                hour * MJD_Time.SECONDS_IN_HOUR
                + minute * MJD_Time.SECONDS_IN_MINUTE
                + second
            )

            self.__add__(daySeconds)

        elif all(v is not None for v in [year, day, second]):
            initialize_year_day_second(year, day, second)

        elif day is not None:
            self.day = getFloor(day)
            self.second = float((day - self.day) * MJD_Time.SECONDS_IN_DAY)

        else:
            self.day = int(d)
            self.second = float(s)

            self.normalize()

    def get_Julian_Century(self):
        dTTDays = (self.day - J2000) + (
            self.second + MJD_Time.TAI_GPS_OFFSET + TT_TAI_SECS
        ) / MJD_Time.SECONDS_IN_DAY
        return dTTDays / MJD_Time.JULIAN_CENTURY

    def get_UT1_Time(self, dUT1_TAI, day=None, frac_day=None):
        if day is not None and frac_day is not None:
            siDu = self.day - math.floor(J2000)
            dFu = (
                self.second + MJD_Time.TAI_GPS_OFFSET + dUT1_TAI
            ) / MJD_Time.SECONDS_IN_DAY - (J2000 - math.floor(J2000))
            if dFu < 0.0:
                siDu -= 1
                dFu += 1.0

            return siDu, dFu

        else:
            x = 0
            y = 0.0
            x, y = self.get_UT1_Time(dUT1_TAI, x, y)
            return float(x) + y

    def get_ModifiedJulianDay(self):
        dTTSecs = self.second + MJD_Time.TAI_GPS_OFFSET + TT_TAI_SECS
        return self.day + dTTSecs / MJD_Time.SECONDS_IN_DAY

    def get_JulianDay_TT(self):
        dTTSecs = self.second + MJD_Time.TAI_GPS_OFFSET + TT_TAI_SECS
        return MJD_TO_JD + self.day + dTTSecs / MJD_Time.SECONDS_IN_DAY

    def get_years(self):
        years = (
            (self.day - J2000)
            + (self.second + MJD_Time.TAI_GPS_OFFSET) / MJD_Time.SECONDS_IN_DAY
        ) / MJD_Time.DAYS_IN_YEAR
        return years

    def get_day_of_year(self):
        _, _, dayOfYear = self.get_year_and_day()
        return dayOfYear

    def get_year_and_day(self):
        mjd_day_offset = self.day - MJD_1980

        remaining_days = mjd_day_offset
        year_index = YEAR_1980
        while remaining_days >= 0:
            num_days_in_year = (
                MJD_Time.DAYS_IN_YEAR + 1
                if MJD_Time.isLeapYear(year_index)
                else MJD_Time.DAYS_IN_YEAR
            )

            if remaining_days < num_days_in_year:
                int_day_of_year = remaining_days + 1
                double_day_of_year = int_day_of_year + (
                    self.second / MJD_Time.SECONDS_IN_DAY
                )
                return year_index, int_day_of_year, double_day_of_year

            remaining_days -= num_days_in_year
            year_index += 1

    def get_calendar_date(self):
        year, int_day_of_year, _ = self.get_year_and_day()
        is_leap_year = MJD_Time.isLeapYear(year)

        month_index = 1
        day_of_month = int_day_of_year
        num_month_days = DAYS_IN_MONTH[1]
        while day_of_month > num_month_days:
            day_of_month -= num_month_days
            month_index += 1
            num_month_days = DAYS_IN_MONTH[month_index]
            if month_index == 2 and is_leap_year:
                num_month_days += 1

        hour = minute = m_second = 0
        if self.second >= 0:
            day_secs = int(self.second)
            hour = day_secs // MJD_Time.SECONDS_IN_HOUR
            hour_secs = day_secs % MJD_Time.SECONDS_IN_HOUR
            minute = hour_secs // MJD_Time.SECONDS_IN_MINUTE
            m_second = day_secs % MJD_Time.SECONDS_IN_MINUTE

        m_second_float = m_second + (self.second - getFloor(self.second))

        return year, month_index, day_of_month, hour, minute, m_second, m_second_float

    def add_seconds(self, seconds):
        self.second += seconds
        self.normalize()

    def add_day(self, day):
        self.day += day

    def normalize(self):
        if self.second >= 0 and self.second < MJD_Time.SECONDS_IN_DAY:
            return

        dayFrac = self.second / MJD_Time.SECONDS_IN_DAY
        intDayFrac = getFloor(dayFrac)

        self.day += intDayFrac
        self.second -= intDayFrac * MJD_Time.SECONDS_IN_DAY

    @staticmethod
    def isLeapYear(year):
        return False if (year % 4 or not year % 100) and (year % 400) else True

    def __iadd__(self, other):
        self.second += other
        self.normalize()
        return self

    def __isub__(self, other):
        self.second -= other
        self.normalize()
        return self

    def __add__(self, rhs):
        self.__iadd__(rhs)
        return self

    def __sub__(self, rhs):
        if isinstance(rhs, MJD_Time):
            dayOffset = self.day - rhs.day
            secOffset = self.second - rhs.second

            offset = dayOffset * MJD_Time.SECONDS_IN_DAY + secOffset

            return offset
        else:
            self.__isub__(rhs)
            return self

    def __eq__(self, other):
        return math.fabs(self.__sub__(other)) < COMP_THRES

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.__sub__(other) < -COMP_THRES

    def __gt__(self, other):
        return self.__sub__(other) > COMP_THRES

    def __le__(self, other):
        return self.__sub__(other) < COMP_THRES

    def __ge__(self, other):
        return self.__sub__(other) > -COMP_THRES

    def __str__(self):
        return f"({self.day}, {self.second})"

    def __repr__(self):
        return f"MJD_Time(day={self.day}, second={self.second})"


class GPS_Time:
    SECONDS_IN_DAY = 86400
    DAYS_IN_WEEK = 7
    SECONDS_IN_WEEK = SECONDS_IN_DAY * DAYS_IN_WEEK
    GPS_UNIX_OFFSET = 315964800
    SECONDS_IN_HOUR = 3600
    SECONDS_IN_QUARTER = 900
    SECONDS_IN_MINUTE = 60
    DAYS_IN_YEAR_I = 365
    TAI_GPS_OFFSET = 19
    JULIAN_CENTURY = 36525.0
    DAYS_IN_YEAR_D = 365.25
    MOSCOW_HOUR_UTC_OFFSET = 3 * SECONDS_IN_HOUR
    GPS_BDS_OFFSET = 14
    GPS_EGNOS_OFFSET = 619315200

    second = 0.0
    week = 0

    leap_seconds_ = 0
    leap_seconds_table_ = {}

    @staticmethod
    def set_UTC_TAI(utc_tai, date=None):
        if isinstance(utc_tai, int):
            GPS_Time.leap_seconds_ = utc_tai - MJD_Time.TAI_GPS_OFFSET

        elif isinstance(utc_tai, dict):
            for date, utc_tai in utc_tai.items():
                GPS_Time.set_UTC_TAI(date=date, utc_tai=utc_tai)

        elif isinstance(date, GPS_Time):
            GPS_Time.leap_seconds_table_[date] = utc_tai - MJD_Time.TAI_GPS_OFFSET

    @staticmethod
    def get_leap_seconds(date=None):
        if isinstance(date, GPS_Time):
            if not bool(GPS_Time.leap_seconds_table_):
                return GPS_Time.leap_seconds_
            else:
                keys = sorted(GPS_Time.leap_seconds_table_)
                for i, key in enumerate(keys):
                    if key > date:
                        if i == 0:
                            return GPS_Time.leap_seconds_table_[key]
                        else:
                            return GPS_Time.leap_seconds_table_[keys[i - 1]]

        else:
            if not GPS_Time.leap_seconds_table_:
                return GPS_Time.leap_seconds_
            else:
                return list(GPS_Time.leap_seconds_table_.values())[-1]

    @staticmethod
    def set_leap_seconds(leap_seconds, date=None):
        if isinstance(leap_seconds, int):
            GPS_Time.leap_seconds_ = leap_seconds

        elif isinstance(leap_seconds, dict):
            for date, leap_seconds in leap_seconds.items():
                GPS_Time.set_leap_seconds(date=date, leap_seconds=leap_seconds)

        elif isinstance(date, GPS_Time):
            GPS_Time.leap_seconds_table_[date] = leap_seconds

    @staticmethod
    def leap_seconds_value_valid():
        return bool(GPS_Time.leap_seconds_table_) or (GPS_Time.leap_seconds_ != 0)

    @staticmethod
    def now():
        return GPS_Time(unix_seconds=int(time.time()))

    @staticmethod
    def now_microsec():
        now = datetime.now(timezone.utc)
        unix_seconds = int(now.timestamp())

        return GPS_Time(unix_seconds + now.microsecond * 1e-6)

    @staticmethod
    def inf():
        return GPS_Time(year=2099, doy=1, second=0)

    @staticmethod
    def year_2d_to_4d(year):
        if year > 80:
            return year + 1900
        return year + 2000

    @staticmethod
    def get_microseconds_fraction(seconds):
        return round(seconds * 1e6) - int(seconds) * 1e6

    def __init__(
        self,
        w=0,
        s=0,
        rhs=None,
        year=None,
        month=None,
        day=None,
        doy=None,
        hour=None,
        minute=None,
        second=None,
        mjd_time=None,
        unix_seconds=None,
    ):
        def initialize_from_mjd(mjd: MJD_Time):
            siDaysOffset = mjd.day - GST_TIME_START_MJD_DAYS
            self.week = int(siDaysOffset // GPS_Time.DAYS_IN_WEEK)

            siWeekDays = siDaysOffset % GPS_Time.DAYS_IN_WEEK
            self.second = float(siWeekDays * GPS_Time.SECONDS_IN_DAY + mjd.second)

            self.normalize()

        if isinstance(mjd_time, MJD_Time):
            initialize_from_mjd(mjd_time)

        elif isinstance(unix_seconds, (float, int)):
            t = unix_seconds - GPS_Time.GPS_UNIX_OFFSET

            tmp_week = t / GPS_Time.SECONDS_IN_WEEK
            tmp_second = t % GPS_Time.SECONDS_IN_WEEK

            temp = GPS_Time(tmp_week, tmp_second)
            t += self.get_leap_seconds(temp)

            self.week = int(t / GPS_Time.SECONDS_IN_WEEK)
            self.second = float(t % GPS_Time.SECONDS_IN_WEEK)

        elif isinstance(rhs, GPS_Time):
            if self.__ne__(rhs):
                self.week = int(rhs.week)
                self.second = rhs.second

        elif all(v is not None for v in [year, month, day, hour, minute, second]):
            mjdTime = MJD_Time(
                year=year, month=month, day=day, hour=hour, minute=minute, second=second
            )

            initialize_from_mjd(mjdTime)

        elif all(v is not None for v in [year, doy, hour, minute, second]):
            daySeconds = (
                hour * GPS_Time.SECONDS_IN_HOUR
                + minute * GPS_Time.SECONDS_IN_MINUTE
                + second
            )
            mjdTime = MJD_Time(year=year, doy=doy, second=daySeconds)

            initialize_from_mjd(mjdTime)

        elif all(v is not None for v in [year, doy, second]):
            mjdTime = MJD_Time(year=year, doy=doy, second=second)

            initialize_from_mjd(mjdTime)

        else:
            self.second = float(s)
            self.week = int(w)

            self.normalize()

    def __copy__(self):
        return type(self)(self.week, self.second)

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self.week, memo), copy.deepcopy(self.second, memo)
        )

    def day_sec(self):
        temp = self.get_UNIX_seconds() + GPS_Time.get_leap_seconds(self)
        return datetime.fromtimestamp(temp, timezone.utc).second

    def min(self):
        temp = self.get_UNIX_seconds() + GPS_Time.get_leap_seconds(self)
        return datetime.fromtimestamp(temp, timezone.utc).minute

    def hour(self):
        temp = self.get_UNIX_seconds() + GPS_Time.get_leap_seconds(self)
        return datetime.fromtimestamp(temp, timezone.utc).hour

    def day(self):
        temp = self.get_UNIX_seconds() + GPS_Time.get_leap_seconds(self)
        return datetime.fromtimestamp(temp, timezone.utc).day

    def month(self):
        temp = self.get_UNIX_seconds() + GPS_Time.get_leap_seconds(self)
        return datetime.fromtimestamp(temp, timezone.utc).month

    def year(self):
        temp = self.get_UNIX_seconds() + GPS_Time.get_leap_seconds(self)
        return datetime.fromtimestamp(temp, timezone.utc).year

    def doy(self):
        temp = self.get_UNIX_seconds() + GPS_Time.get_leap_seconds(self)
        return datetime.fromtimestamp(temp, timezone.utc).timetuple().tm_yday

    def day_sec_UTC(self):
        temp = self.get_UNIX_seconds()
        return datetime.fromtimestamp(temp, timezone.utc).second

    def min_UTC(self):
        temp = self.get_UNIX_seconds()
        return datetime.fromtimestamp(temp, timezone.utc).minute

    def hour_UTC(self):
        temp = self.get_UNIX_seconds()
        return datetime.fromtimestamp(temp, timezone.utc).hour

    def day_UTC(self):
        temp = self.get_UNIX_seconds()
        return datetime.fromtimestamp(temp, timezone.utc).day

    def month_UTC(self):
        temp = self.get_UNIX_seconds()
        return datetime.fromtimestamp(temp, timezone.utc).month

    def year_UTC(self):
        temp = self.get_UNIX_seconds()
        return datetime.fromtimestamp(temp, timezone.utc).year

    def doy_UTC(self):
        temp = self.get_UNIX_seconds()
        return datetime.fromtimestamp(temp, timezone.utc).timetuple().tm_yday

    def get_UNIX_seconds(self):
        return (
            self.get_GPS_abs_seconds()
            + GPS_Time.GPS_UNIX_OFFSET
            - GPS_Time.get_leap_seconds(self)
        )

    def get_GPS_abs_seconds(self):
        return self.week * GPS_Time.SECONDS_IN_WEEK + self.second

    def get_day_of_year(self):
        mjdTime = MJD_Time(time=self)
        return mjdTime.get_day_of_year()

    def get_year_and_day(self):
        mjdTime = MJD_Time(time=self)
        return mjdTime.get_year_and_day()

    def get_calendar_date(self):
        mjdTime = MJD_Time(time=self)
        return mjdTime.get_calendar_date()

    def get_Julian_Century(self):
        mjdTime = MJD_Time(time=self)
        return mjdTime.get_Julian_Century()

    def get_UT1_Time(self, dUT1_TAI, day=None, frac_day=None):
        mjdTime = MJD_Time(time=self)
        if isinstance(day, int) and isinstance(frac_day, float):
            return mjdTime.get_UT1_Time(dUT1_TAI, day, frac_day)
        else:
            return mjdTime.get_UT1_Time(dUT1_TAI)

    def get_ModifiedJulianDay(self):
        mjdTime = MJD_Time(time=self)
        return mjdTime.get_ModifiedJulianDay()

    def get_JulianDay_TT(self):
        mjdTime = MJD_Time(time=self)
        return mjdTime.get_JulianDay_TT()

    def get_years(self):
        mjdTime = MJD_Time(time=self)
        return mjdTime.get_years()

    def add_seconds(self, seconds):
        self.second += seconds
        self.normalize()

    def add_week(self, week):
        self.week += week

    def normalize(self):
        if self.second >= 0 and self.second < GPS_Time.SECONDS_IN_WEEK:
            return

        weekFrac = self.second / GPS_Time.SECONDS_IN_WEEK
        intWeekFrac = getFloor(weekFrac)

        self.week += intWeekFrac
        self.second -= intWeekFrac * GPS_Time.SECONDS_IN_WEEK

    def round_time(self):
        aux = GPS_Time()
        aux.week = self.week
        aux.second = getInt(self.second)

        aux.normalize()

        return aux

    def reset(self):
        self.week = int(0)
        self.second = float(0.0)

    def calendar_text_str(self, add_leap_seconds=True):
        unix_seconds = self.get_UNIX_seconds()
        if add_leap_seconds:
            unix_seconds += GPS_Time.get_leap_seconds(self)

        utc_time = datetime.fromtimestamp(unix_seconds, timezone.utc)
        return utc_time.strftime("%x %X")

    def calendar_text_str_d(self, add_leap_seconds=True):
        base_str = self.calendar_text_str(add_leap_seconds)
        microseconds = GPS_Time.get_microseconds_fraction(self.second)
        return f"{base_str}.{microseconds:06.0f}"

    def calendar_column_str(self):
        year, month, day, hour, minute, m_second, _ = self.get_calendar_date()
        return (
            f"{year:4}  {month:2}  {day:2}  {hour:2}  {minute:2}  {getInt(m_second):2}"
        )

    def calendar_column_str_d(self):
        year, month, day, hour, minute, _, m_second = self.get_calendar_date()
        return f"{year:4}  {month:2}  {day:2}  {hour:2}  {minute:2}  {m_second:9.6f}"

    def DB_str(self, add_leap_seconds=True):
        unix_seconds = self.get_UNIX_seconds()
        if add_leap_seconds:
            unix_seconds += GPS_Time.get_leap_seconds(self)
        utc_time = datetime.fromtimestamp(unix_seconds, timezone.utc)
        return utc_time.strftime("%Y-%m-%d %H:%M:%S")

    def canon_str(self):
        return f"({self.week:02},{self.second:7.2f})"

    def __iadd__(self, other):
        self.second += other
        self.normalize()
        return self

    def __isub__(self, other):
        self.second -= other
        self.normalize()
        return self

    def __add__(self, rhs):
        self.__iadd__(rhs)
        return self

    def __sub__(self, rhs):
        if isinstance(rhs, GPS_Time):
            weekOffset = self.week - rhs.week
            secOffset = self.second - rhs.second

            offset = weekOffset * GPS_Time.SECONDS_IN_WEEK + secOffset

            return offset
        else:
            self.__isub__(rhs)
            return self

    def __eq__(self, other):
        return (
            math.fabs(self.get_GPS_abs_seconds() - other.get_GPS_abs_seconds())
            < COMP_THRES
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return (
            self.get_GPS_abs_seconds() < other.get_GPS_abs_seconds()
        ) and not self.__eq__(other)

    def __gt__(self, other):
        return (
            self.get_GPS_abs_seconds() > other.get_GPS_abs_seconds()
        ) and not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __str__(self):
        return f"({self.week}, {self.second})"

    def __repr__(self):
        return f"GPS_Time(week={self.week}, second={self.second})"

    def __hash__(self):
        rounded_second = round(self.second, int(-math.log10(COMP_THRES)))
        return hash((self.week, rounded_second))
