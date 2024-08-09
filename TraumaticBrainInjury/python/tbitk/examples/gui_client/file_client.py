import argparse
import itk
import pyigtl
import numpy as np
import tbitk.util as util
import numpy as np
from pathlib import Path

def message_from_image(img, device_name):
    ijk_to_world_matrix = np.eye(4)
    ijk_to_world_matrix[0,0] = img.GetSpacing()[0]
    ijk_to_world_matrix[1,1] = img.GetSpacing()[1]
    return pyigtl.ImageMessage(itk.array_from_image(img), ijk_to_world_matrix=ijk_to_world_matrix, device_name=device_name)



def _construct_parser():
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument(
        "--device_name",
        action="store",
        required=True,
        help="Device name from which the messages will be sent"
    )

    my_parser.add_argument(
        "--source_path",
        action="store",
        type=Path,
        required=True,
        help="Source path to run the gui and inference on"
    )

    my_parser.add_argument(
        "--port",
        action="store",
        type=int,
        default=18944,
    )

    my_parser.add_argument(
        "--act_as_client",
        action="store_true",
        help="Whether or not the gui should act like a client instead of a server"
    )

    my_parser.add_argument(
        "--host",
        action="store",
        type=str,
        default="127.0.0.1"
    )

    my_parser.add_argument(
        "--continuous",
        action="store_true",
        help="Loop over the video, sending frames indefinitely"
    )

    return my_parser


if __name__ == "__main__":
    my_parser = _construct_parser()
    args = my_parser.parse_args()
    print("Reading image at path", str(args.source_path))
    img = itk.imread(str(args.source_path.resolve()))
    print("Image read successfully. Starting client")

    if args.act_as_client:
        conn = pyigtl.OpenIGTLinkClient(args.host, args.port)
    else:
        conn = pyigtl.OpenIGTLinkServer(args.port)
    client_or_host_str = "client" if args.act_as_client else "server"
    print(client_or_host_str, "started successfully")

    input('Press Enter to send images.')

    while True:
        for i in range(img.GetLargestPossibleRegion().GetSize()[2]):
            print('Sending slice #' + str(i))
            msg = message_from_image(util.extract_slice(img, i), device_name=args.device_name)
            conn.send_message(msg, wait=False)

        if not args.continuous:
            break

    input('Press Enter to close.')
    conn.stop()
