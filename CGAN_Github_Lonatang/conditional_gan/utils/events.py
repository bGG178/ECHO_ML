# Copyright (c) AlphaBetter. All rights reserved.
import logging
import os
import shutil
from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    "configure_logging", "LOGGER", "NCOLS", "Summary", "AverageMeter", "ProgressMeter",
]


def configure_logging(name: Optional[str] = None) -> logging.Logger:
    """Configure the logging system.

    Args:
        name (Optional[str], optional): Logger name. Defaults to None.

    Returns:
        logging.Logger: Logger.
    """
    logger = logging.getLogger(name)
    rank = int(os.getenv("RANK", -1))

    # Set the logging level
    logger_level = logging.INFO if (rank in (-1, 0)) else logging.WARNING
    logger.setLevel(logger_level)

    # Set the format of the log
    format_string = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Print logs to console
    cli_handler = logging.StreamHandler()
    cli_handler.setFormatter(format_string)
    cli_handler.setLevel(logger_level)
    logger.handlers.clear()
    logger.addHandler(cli_handler)

    return logger


class Summary(Enum):
    """Enumeration for the type of summary."""
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name: str, fmt: str = ":f", summary_type: Summary = Summary.AVERAGE) -> None:
        """Initialize the AverageMeter object.

        Args:
            name (str): The name of the metric being tracked.
            fmt (str): The format string for displaying the values. Default is ":f".
            summary_type (Summary): The type of summary to use (e.g., average, sum). Default is Summary.AVERAGE.
        """
        self.name: str = name
        self.fmt: str = fmt
        self.summary_type: Summary = summary_type
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to their initial values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the statistics with a new value.

        Args:
            val (float): The new value to add.
            n (int): The weight of the value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # Calculate the new average

    def all_reduce(self) -> None:
        """Perform an all-reduce operation to aggregate values across devices in distributed training."""
        # Determine the device to perform the operation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Create a tensor to store the sum and count
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        # Perform the all-reduce operation (summing values across devices)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        # Update the statistics with the aggregated results
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """Format the current and average values as a string.

        Returns:
            str: Formatted string representing the metric values.
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        """Generate a summary of the metric based on the specified summary type.

        Returns:
            str: The formatted summary string.
        """
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("Invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches: int, meters: list[AverageMeter], prefix: str = "") -> None:
        """Initialize the ProgressMeter.

        Args:
            num_batches (int): Total number of batches.
            meters (list[AverageMeter]): List of AverageMeter objects to track.
            prefix (str): Prefix string for display (default is "").
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """Display the current progress for a specific batch.

        Args:
            batch (int): The current batch number.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self) -> None:
        """Display a summary of all tracked metrics."""
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        """Generate a format string for batch numbers.

        Args:
            num_batches (int): Total number of batches.

        Returns:
            str: Format string for displaying batch numbers.
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


LOGGER = configure_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)
