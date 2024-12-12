import argparse
import os
import shutil
import signal
import sys

from navutils.logger import Logger
from navutils.user_interrupt import UserInterruptException, signal_handler
from pewrapper.misc import RELEASE_INFO, parse_session_file
from pewrapper.wrapper_handler import Wrapper_Handler

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def main(argv):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ################################################
    ###############  INPUTS SECTION  ###############
    ################################################
    parser = argparse.ArgumentParser(description="Allowed options")
    parser.version = RELEASE_INFO
    parser.add_argument("-v", "--version", help="print version", action="version")
    parser.add_argument(
        "-s", "--session_file", help="session file", dest="session_file_path"
    )
    parser.add_argument(
        "-o", "--output_directory", help="output directory", dest="output_path"
    )
    parser.add_argument(
        "-g", "--debug_level", help="debug level (INFO/WARNING/ERROR/DEBUG)"
    )
    parser.add_argument(
        "-p", "--parsing_rate", type=int, default=0, help="parsing_rate(optional)"
    )

    args = parser.parse_args(argv[1:])
    args.output_path = os.path.join(args.output_path)
    Logger(args.output_path)

    if not args.output_path or not args.debug_level:
        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.MAIN,
            f"Not enough parameters. The command line shall be: {argv[0]} --session_file --output_directory --debug_level(INFO/WARNING/ERROR/DEBUG/TRACE) --parsing_rate(optional)",
        )
        parser.print_help()
        return EXIT_FAILURE

    Logger.set_category(args.debug_level)

    if args.parsing_rate:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f" Parsing rate set at {args.parsing_rate}",
        )
    else:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f" Parsing rate input not received. Epochs are processed without rate filter",
        )

    ####################################################
    #################  INITIALIZATION  #################
    ####################################################

    output_path = os.path.join(args.output_path, "")

    if not (result := parse_session_file(args.session_file_path))[0]:
        _, _, _, _, addInfo_session, _, _ = result
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.READER,
            f"Error processing session file: {addInfo_session}",
        )
        Logger.reset()
        return EXIT_FAILURE
    (
        _,
        config_file_path,
        wrapper_file_path,
        tracing_config_file,
        _,
        initial_epoch_session,
        final_epoch_session,
    ) = result

    # COPY INPUT FILES TO OUTPUT FOLDER
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Session Config file
    session_pe_file_path = args.session_file_path
    session_file_destination = os.path.join(output_path, "session_output_PE.txt")
    if os.path.isfile(session_pe_file_path):
        shutil.copyfile(session_pe_file_path, session_file_destination)
    else:
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.MAIN,
            f"Session file path: {session_pe_file_path} does not exist. Copy to output folder aborted",
        )

    # Wrapper Config file
    wrapper_config_file_path = config_file_path
    wrapper_config_file_destination = os.path.join(output_path, "wrapper_config.txt")
    if os.path.isfile(wrapper_config_file_path):
        shutil.copyfile(wrapper_config_file_path, wrapper_config_file_destination)
    else:
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.MAIN,
            f"Wrapper Configuration file path: {wrapper_config_file_path} does not exist. Copy to output folder aborted",
        )

    # Tracing config file
    wrapper_tracing_file_path = tracing_config_file
    wrapper_tracing_file_destination = os.path.join(output_path, "tracing_config.txt")
    if os.path.isfile(wrapper_tracing_file_path):
        shutil.copyfile(wrapper_tracing_file_path, wrapper_tracing_file_destination)
    else:
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.MAIN,
            f"Tracing Configuration file path: {wrapper_tracing_file_path} does not exist. Copy to output folder aborted",
        )

    ################################################
    #################  PROCESSING  #################
    ################################################

    wrapper_handler = Wrapper_Handler(
        args.debug_level,
        tracing_config_file,
        output_path,
        initial_epoch_session,
        final_epoch_session,
    )
    return_value = EXIT_FAILURE

    try:
        if wrapper_handler.process_scenario(
            config_file_path, wrapper_file_path, output_path, args.parsing_rate
        ):
            return_value = EXIT_SUCCESS

    except UserInterruptException as exception:
        return_value = EXIT_FAILURE
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"{exception.get_msg()}",
        )

    if not wrapper_handler.close_PE():
        return_value = EXIT_FAILURE
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"Error closing files of PE: ",
        )

    else:
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"Tracing files have been closed",
        )

    Logger.reset()

    return return_value


if __name__ == "__main__":
    sys.exit(main(sys.argv))
