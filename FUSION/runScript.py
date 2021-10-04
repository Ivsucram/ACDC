from multistream import main

"""
This script is for calling the main function from multistream.py. There are two parameters,
- First parameter is the file location (Without the extension. Extension can be specified from config.properties)
- Second parameter is the percentage of data the source stream receives. As an example, if this parameter is 0.1,
 it means that randomly 10% of incoming instances go to the source stream, and rest 90% go to the target stream.
"""

print "Running FC"
main('usps_mnist', 0.1)
print "Done"
