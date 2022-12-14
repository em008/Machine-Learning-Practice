import pandas as pd
import time
import ast

def replace_address(row):
    if "Netherlands" in row["Hotel_Address"]:
        return "Amsterdam, Netherlands"
    elif "Barcelona" in row["Hotel_Address"]:
        return "Barcelona, Spain"
    elif "United Kingdom" in row["Hotel_Address"]:
        return "London, United Kingdom"
    elif "Milan" in row["Hotel_Address"]:        
        return "Milan, Italy"
    elif "France" in row["Hotel_Address"]:
        return "Paris, France"
    elif "Vienna" in row["Hotel_Address"]:
        return "Vienna, Austria" 
    else:
        return row.Hotel_Address
    
# Load the hotel reviews from CSV
start = time.time()
df = pd.read_csv('../../data/Hotel_Reviews.csv')

# Dropping columns
df.drop(["lat", "lng", "Additional_Number_of_Scoring"], axis = 1, inplace=True)

# Replace all the addresses with a shortened
df["Hotel_Address"] = df.apply(replace_address, axis = 1)

# Replace `Total_Number_of_Reviews` and `Average_Score` with calculated values
df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)

# Process the Tags into new columns
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

# No longer need these columns
df.drop(["Review_Date", "Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../../data/Hotel_Reviews_Filtered.csv', index = False)
end = time.time()
print("Filtering took " + str(round(end - start, 2)) + " seconds")

# Load the filtered data from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")

# Remove all quotes
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)

# Split the strings into a list
tag_list_df = df.Tags.str.split(',', expand = True)

# Remove leading and trailing spaces
df["Tag_1"] = tag_list_df[0].str.strip()
df["Tag_2"] = tag_list_df[1].str.strip()
df["Tag_3"] = tag_list_df[2].str.strip()
df["Tag_4"] = tag_list_df[3].str.strip()
df["Tag_5"] = tag_list_df[4].str.strip()
df["Tag_6"] = tag_list_df[5].str.strip()

# Merge the columns into one with melt
df_tags = df.melt(value_vars=["Tag_1", "Tag_2", "Tag_3", "Tag_4", "Tag_5", "Tag_6"])

# Get the value counts
tag_vc = df_tags.value.value_counts()
# print(tag_vc)
print("The shape of the tags with no filtering:", str(df_tags.shape))

# Drop rooms, suites, and length of stay, mobile device and anything with less count than a 1000
df_tags = df_tags[~df_tags.value.str.contains("Standard|room|Stayed|device|Beds|Suite|Studio|King|Superior|Double", na=False, case=False)]
tag_vc = df_tags.value.value_counts().reset_index(name="count").query("count > 1000")

# Print the top 10
print(tag_vc[:10])
