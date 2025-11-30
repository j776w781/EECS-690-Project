import cv2
import os
import argparse
import PinDetector
import read_ic_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('mark_text')
    args = parser.parse_args()

    if args.mark_text not in ['0', '1']:
        raise Exception("mark_text arg must be 0 or 1")


    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)


    marked_pins = PinDetector.mark_pins(args.input)
    
    if args.mark_text == "0":
        output_img = marked_pins
    else:
        output_img = read_ic_text.read_ic_text(marked_pins)
    
    
    cv2.imwrite(args.output, output_img)



if __name__ == "__main__":
    main()