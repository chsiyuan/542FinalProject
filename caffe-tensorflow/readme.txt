Only for converting .caffemodle to .npy:

main script: convert.py in the root dir 
usage example: python convert.py def_path examples/test.prototxt --caffemodel examples/VGG16.caffemodel --data-output-path ./VGG16.npy -p test(train or test)

sample prototxt: ./examples/test.prototxt

Not sure if it can be done without qsub...
