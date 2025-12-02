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

    if args.mark_text not in ['0', '1', '2']:
        raise Exception("mark_text arg must be 0, 1, or 2")


    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)


    #marked_pins = PinDetector.mark_pins(args.input)
    
    if args.mark_text == "0":
        output_img = PinDetector.mark_pins(args.input)
    elif args.mark_text == "1":
        img = cv2.imread(args.input)
        output_img = read_ic_text.read_ic_text(img)
    else:
        output_img = read_ic_text.read_ic_text(PinDetector.mark_pins(args.input))
    
    
    cv2.imwrite(args.output, output_img)



if __name__ == "__main__":
    main()