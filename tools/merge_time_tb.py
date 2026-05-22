import os
import glob
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.summary.writer.record_writer import RecordWriter
import argparse

def merge_actual_time(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "events.out.tfevents.0000000000.merged")
    writer = RecordWriter(open(out_file, "wb"))

    files = sorted(glob.glob(os.path.join(input_dir, "events.out.tfevents.*")))
    if not files:
        print(f"No event files found in {input_dir}")
        return

    accumulated_gap = 0.0
    last_real_wall_time = None

    for file in files:
        print(f"Processing {os.path.basename(file)}...")
        loader = RawEventFileLoader(file)
        
        file_first_event = True
        for raw_event in loader.Load():
            try:
                event = Event.FromString(raw_event)
            except Exception as e:
                continue
                
            if last_real_wall_time is None:
                last_real_wall_time = event.wall_time
                
            if file_first_event:
                # Calculate gap from last session to this session's first event
                gap = event.wall_time - last_real_wall_time
                if gap > 0:
                    # Keep a small 1-second gap to continuously connect sessions
                    accumulated_gap += (gap - 1.0)
                file_first_event = False
                
            last_real_wall_time = event.wall_time
            
            # Adjust event time
            event.wall_time = event.wall_time - accumulated_gap
            writer.write(event.SerializeToString())

    writer.flush()
    writer.close()
    print(f"Merged output saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge TensorBoard event files removing large time gaps (offline time).")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing event files")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory to save merged event file")
    args = parser.parse_args()
    
    merge_actual_time(args.input, args.output)
