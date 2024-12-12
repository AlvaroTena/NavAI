from typing import Union

from navutils.logger import Logger
from pewrapper.api import (
    Configuration_info,
    PE_Output_str,
    Position_Engine_API,
    SafeState,
    SafeStateMachineSignal,
    SystemStatus,
)
from pewrapper.types import GPS_Time


class OutputStr:
    def __init__(self):
        self.output_PE = PE_Output_str()
        self.SafeState = SafeState.INACTIVE_STATE

    def reset(self):
        self.__init__()


LOGSIZ = 1024


def GetSignalString(SSM_signal: SafeStateMachineSignal) -> str:
    if SSM_signal == SafeStateMachineSignal.FATAL_ERROR:
        signal = "FATAL_ERROR STATE"
    elif SSM_signal == SafeStateMachineSignal.RESET_FILTER:
        signal = "RESET_FILTER STATE"
    elif SSM_signal == SafeStateMachineSignal.INIT_MON:
        signal = "INITIAL_MON STATE"
    elif SSM_signal == SafeStateMachineSignal.CHECK_FAIL_ALGO:
        signal = "CHECK_FAIL_ALGO STATE"
    elif SSM_signal == SafeStateMachineSignal.CHECK_OK_ALGO:
        signal = "CHECK_OK_ALGO STATE"
    elif SSM_signal == SafeStateMachineSignal.NO_SOLUTION:
        signal = "NO_SOLUTION STATE"
    elif SSM_signal == SafeStateMachineSignal.NON_SAFE_CONDITIONS:
        signal = "NON_SAFE_CONDITIONS STATE"
    elif SSM_signal == SafeStateMachineSignal.SAFE_CONDITIONS:
        signal = "SAFE_CONDITIONS STATE"
    elif SSM_signal == SafeStateMachineSignal.START_SIGNAL:
        signal = "START_SIGNAL"
    elif SSM_signal == SafeStateMachineSignal.TERMINATE_SIGNAL:
        signal = "TERMINATE_SIGNAL"
    else:
        signal = "SIGNAL N/A"
    return signal


class SafetyStateMachine:
    def __init__(
        self,
        position_engine: Position_Engine_API,
        config_info: Configuration_info,
    ):
        self.current_state_ = SafeState.INACTIVE_STATE
        self.position_engine_ = position_engine
        self.config_info_ = config_info

    def reset(self):
        self.current_state_ = SafeState.INACTIVE_STATE
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.STATE_MACHINE,
            "State Transition: Entering PE_API::INACTIVE STATE",
        )

        return True

    def ProcessSignal(
        self,
        SSM_signal: Union[OutputStr, SafeStateMachineSignal],
    ) -> bool:
        def ProcessSignalSafeStateMachineSignal(signal):
            result = True

            if self.current_state_ == SafeState.INIT_STATE:
                result &= self.DoInitStateTransition(signal)
            elif self.current_state_ == SafeState.NO_SOLUTION_STATE:
                result &= self.DoNoSolutionStateTransition(signal)
            elif self.current_state_ == SafeState.INIT_MON_STATE:
                result &= self.DoInitSolutionStateTransition(signal)
            elif self.current_state_ == SafeState.VALID_SOLUTION_STATE:
                result &= self.DoValidSolutionStateTransition(signal)
            elif self.current_state_ == SafeState.SAFE_SOLUTION_STATE:
                result &= self.DoSafeSolutionStateTransition(signal)
            elif self.current_state_ == SafeState.INACTIVE_STATE:
                result &= self.DoInactiveStateTransition(signal)
            elif self.current_state_ == SafeState.ERROR_STATE:
                result &= self.DoErrorStateTransition(signal)
            else:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.STATE_MACHINE,
                    "Unexpected state",
                )
                result &= self.HandleErrorTransition()
                result &= False

            return result

        if isinstance(SSM_signal, OutputStr):
            output_pe = SSM_signal

            result = ProcessSignalSafeStateMachineSignal(
                output_pe.output_PE.pe_solution_info.SSM_Signal
            )

            result &= self.VacateInvalidPosOutput(output_pe.output_PE)

            output_pe.SafeState = self.current_state_

            return result

        elif isinstance(SSM_signal, SafeStateMachineSignal):
            return ProcessSignalSafeStateMachineSignal(SSM_signal)

    def GetCurrentState(self) -> SafeState:
        return self.current_state_

    def HandleInitTransition(self) -> bool:
        result = True

        if self.current_state_ == SafeState.INIT_STATE:
            self.current_state_ = SafeState.NO_SOLUTION_STATE
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "Signal CHECK_OK_ALGO received. State Transition: INIT -> NO_SOLUTION",
            )

        else:
            result &= self.HandleErrorTransition()
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "Unexpected state reached during processing",
            )
            result &= False

        return result

    def HandleErrorTransition(self) -> bool:
        result = True

        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.STATE_MACHINE,
            "ERROR State reached",
        )
        self.current_state_ = SafeState.ERROR_STATE

        return result

    def HandleValidSolutionTransition(self) -> bool:
        result = True

        if (
            self.current_state_ == SafeState.INIT_MON_STATE
            or self.current_state_ == SafeState.SAFE_SOLUTION_STATE
        ):
            self.current_state_ = SafeState.VALID_SOLUTION_STATE

        else:
            self.current_state_ = SafeState.ERROR_STATE
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "Unexpected state reached during processing",
            )
            result &= False

        return result

    def HandleNoSolutionTransition(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if (
            self.current_state_ == SafeState.VALID_SOLUTION_STATE
            or self.current_state_ == SafeState.SAFE_SOLUTION_STATE
            or self.current_state_ == SafeState.NO_SOLUTION_STATE
            or self.current_state_ == SafeState.INIT_MON_STATE
        ):
            self.current_state_ = SafeState.NO_SOLUTION_STATE
            self.position_engine_.ResetKalmanFilter(SSM_signal)

        else:
            self.current_state_ = SafeState.ERROR_STATE
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "Unexpected state reached during processing",
            )
            result &= False

        return result

    def HandleSafeSolutionTransition(self) -> bool:
        result = True

        if self.current_state_ == SafeState.INIT_MON_STATE:
            self.current_state_ = SafeState.SAFE_SOLUTION_STATE

        else:
            self.current_state_ = SafeState.ERROR_STATE
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "Unexpected state reached during processing",
            )
            result &= False

        return result

    def HandleRestartNeeded(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if (
            self.current_state_ == SafeState.INIT_MON_STATE
            or self.current_state_ == SafeState.VALID_SOLUTION_STATE
            or self.current_state_ == SafeState.SAFE_SOLUTION_STATE
        ):
            self.current_state_ = SafeState.INIT_MON_STATE
            self.position_engine_.ResetKalmanFilter(SSM_signal)

        else:
            self.current_state_ = SafeState.ERROR_STATE
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "Unexpected state reached during processing",
            )
            result &= False

        return result

    def HandleInitMonTransition(self) -> bool:
        result = True

        if (
            self.current_state_ == SafeState.INIT_MON_STATE
            or self.current_state_ == SafeState.NO_SOLUTION_STATE
        ):
            self.current_state_ = SafeState.INIT_MON_STATE

        else:
            self.current_state_ = SafeState.ERROR_STATE
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "Unexpected state reached during processing",
            )
            result &= False

        return result

    def HandleTerminateSignal(self) -> bool:
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.STATE_MACHINE,
            "TERMINATE_SIGNAL received",
        )
        self.reset()

        return True

    def HandleStartSignal(self) -> bool:
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.STATE_MACHINE,
            "START_SIGNAL received",
        )
        self.current_state_ = SafeState.INIT_STATE

        return True

    def DoErrorStateTransition(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if SSM_signal == SafeStateMachineSignal.TERMINATE_SIGNAL:
            result &= self.HandleTerminateSignal()

        else:
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.STATE_MACHINE,
                f"{GetSignalString(SSM_signal)} signal received, ERROR_STATE kept",
            )

        return result

    def DoInactiveStateTransition(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if SSM_signal == SafeStateMachineSignal.START_SIGNAL:
            result &= self.HandleStartSignal()

        elif SSM_signal == SafeStateMachineSignal.FATAL_ERROR:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "State Transition: INACTIVE -> ERROR",
            )
            result &= self.HandleErrorTransition()

        elif SSM_signal == SafeStateMachineSignal.TERMINATE_SIGNAL:
            result &= self.HandleTerminateSignal()

        else:
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.STATE_MACHINE,
                f"{GetSignalString(SSM_signal)} signal received, ERROR_STATE kept",
            )

        return result

    def DoInitStateTransition(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if SSM_signal == SafeStateMachineSignal.CHECK_OK_ALGO:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: CHECK_OK_ALGO received",
            )
            result &= self.HandleInitTransition()

        elif SSM_signal == SafeStateMachineSignal.CHECK_FAIL_ALGO:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: CHECK_FAIL_ALGO received",
            )
            result &= self.HandleErrorTransition()

        elif SSM_signal == SafeStateMachineSignal.FATAL_ERROR:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "State Transition: NO_SOLUTION -> ERROR",
            )
            result &= self.HandleErrorTransition()

        elif SSM_signal == SafeStateMachineSignal.TERMINATE_SIGNAL:
            result &= self.HandleTerminateSignal()

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                f"State Transition: Unexpected Signal Received {GetSignalString(SSM_signal)} entering ERROR state from INIT_STATE",
            )

        return result

    def DoNoSolutionStateTransition(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if SSM_signal == SafeStateMachineSignal.NO_SOLUTION:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: NO_SOLUTION State kept",
            )
            result &= self.HandleNoSolutionTransition(SSM_signal)

        elif SSM_signal == SafeStateMachineSignal.INIT_MON:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: NO_SOLUTION -> INIT_MON_STATE",
            )
            result &= self.HandleInitMonTransition()

        elif SSM_signal == SafeStateMachineSignal.FATAL_ERROR:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "State Transition: NO_SOLUTION -> ERROR",
            )
            result &= self.HandleErrorTransition()

        elif SSM_signal == SafeStateMachineSignal.TERMINATE_SIGNAL:
            result &= self.HandleTerminateSignal()

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                f"State Transition: Unexpected Signal Received {GetSignalString(SSM_signal)} entering ERROR state from NO_SOLUTION",
            )

        return result

    def DoInitSolutionStateTransition(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if SSM_signal == SafeStateMachineSignal.INIT_MON:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: INIT_MON_STATE State kept",
            )

        elif SSM_signal == SafeStateMachineSignal.RESET_FILTER:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: Restart needed. INIT_MON_STATE State kept",
            )
            result &= self.HandleRestartNeeded(SSM_signal)

        elif SSM_signal == SafeStateMachineSignal.NO_SOLUTION:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: INIT_MON_STATE -> NO_SOLUTION",
            )
            result &= self.HandleNoSolutionTransition(SSM_signal)

        elif SSM_signal == SafeStateMachineSignal.NON_SAFE_CONDITIONS:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: INIT_MON_STATE -> VALID_SOLUTION",
            )
            result &= self.HandleValidSolutionTransition()

        elif SSM_signal == SafeStateMachineSignal.SAFE_CONDITIONS:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: INIT_MON_STATE -> SAFE_CONDITIONS",
            )
            result &= self.HandleSafeSolutionTransition()

        elif SSM_signal == SafeStateMachineSignal.FATAL_ERROR:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "State Transition: INIT_MON_STATE -> ERROR",
            )
            result &= self.HandleErrorTransition()

        elif SSM_signal == SafeStateMachineSignal.TERMINATE_SIGNAL:
            result &= self.HandleTerminateSignal()

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                f"State Transition: Unexpected Signal Received {GetSignalString(SSM_signal)} entering ERROR state from INIT_MON_STATE",
            )

        return result

    def DoValidSolutionStateTransition(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if SSM_signal == SafeStateMachineSignal.NON_SAFE_CONDITIONS:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: VALID_SOLUTION State kept",
            )

        elif SSM_signal == SafeStateMachineSignal.NO_SOLUTION:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: VALID_SOLUTION -> NO_SOLUTION",
            )
            result &= self.HandleNoSolutionTransition(SSM_signal)

        elif SSM_signal == SafeStateMachineSignal.RESET_FILTER:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: Restart needed. VALID_SOLUTION -> INIT_MON_STATE",
            )
            result &= self.HandleRestartNeeded(SSM_signal)

        elif SSM_signal == SafeStateMachineSignal.FATAL_ERROR:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "State Transition: VALID_SOLUTION -> ERROR",
            )
            result &= self.HandleErrorTransition()

        elif SSM_signal == SafeStateMachineSignal.TERMINATE_SIGNAL:
            result &= self.HandleTerminateSignal()

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                f"State Transition: Unexpected Signal Received {GetSignalString(SSM_signal)} entering ERROR state from VALID_SOLUTION",
            )

        return result

    def DoSafeSolutionStateTransition(
        self,
        SSM_signal: SafeStateMachineSignal,
    ) -> bool:
        result = True

        if SSM_signal == SafeStateMachineSignal.SAFE_CONDITIONS:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: SAFE_SOLUTION State kept",
            )

        elif SSM_signal == SafeStateMachineSignal.NON_SAFE_CONDITIONS:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: SAFE_SOLUTION -> VALID_SOLUTION",
            )
            result &= self.HandleValidSolutionTransition()

        elif SSM_signal == SafeStateMachineSignal.NO_SOLUTION:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: SAFE_SOLUTION -> NO_SOLUTION",
            )
            result &= self.HandleNoSolutionTransition(SSM_signal)

        elif SSM_signal == SafeStateMachineSignal.RESET_FILTER:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.STATE_MACHINE,
                "State Transition: Restart needed. SAFE_SOLUTION -> INIT_MON_STATE",
            )
            result &= self.HandleRestartNeeded(SSM_signal)

        elif SSM_signal == SafeStateMachineSignal.FATAL_ERROR:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                "State Transition: SAFE_SOLUTION -> ERROR",
            )
            result &= self.HandleErrorTransition()

        elif SSM_signal == SafeStateMachineSignal.TERMINATE_SIGNAL:
            result &= self.HandleTerminateSignal()

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                f"State Transition: Unexpected Signal Received {GetSignalString(SSM_signal)} entering ERROR state from SAFE_SOLUTION",
            )

        return result

    def VacateInvalidPosOutput(
        self,
        output_PE: PE_Output_str,
    ) -> bool:
        dummy_timestamp = GPS_Time(
            w=output_PE.timestamp_week, s=output_PE.timestamp_second
        )

        if (
            dummy_timestamp != GPS_Time()
            and self.current_state_ != SafeState.VALID_SOLUTION_STATE
            and self.current_state_ != SafeState.SAFE_SOLUTION_STATE
            and self.current_state_ != SafeState.INIT_MON_STATE
            and output_PE.pe_solution_info.systemStatus != SystemStatus.PVT
        ):
            output_PE.pe_pos_out = PE_Output_str.PE_pos_out()
            output_PE.pe_solution_info.systemStatus = SystemStatus.NA_SYSTEM_STATUS

        return True
