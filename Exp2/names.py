import os

# Directory containing your files
directory_path = 'models_distil/sgdd'

# Dictionary to store details
file_details = {}

# Loop through files in the directory
for filename in os.listdir(directory_path):
    print(filename)
    file_path = directory_path+"/"+ filename #os.path.join(directory_path, filename)
    print(file_path)
    
    # Check if it's a file
    if os.path.isfile(file_path):
        # Split the filename into parts using underscores
        parts = filename.split('_')
        print(parts)
        
        # Check if the filename has the expected format
        #if len(parts) == 3:
        #_,dataset, compression, seed,_,_ = parts
        key = (parts[1], 'sgdd', float(parts[2]), int(parts[3].split(".")[0]))
            
        # Add entry to the dictionary
        file_details[key] = file_path

# Print the dictionary
print(file_details)
# for key, value in file_details.items():
#     print(f"{key}: {value}")


