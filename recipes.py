import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import emoji

# Load recipe data
recipes = pd.read_csv(r'E:\reciperecom\recipes.csv')

# Preprocess d  ata
recipes['ingredients'] = recipes['ingredients'].fillna('')  # Fill missing values with empty string

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Vectorize ingredients
ingredient_matrix = vectorizer.fit_transform(recipes['ingredients'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(ingredient_matrix, ingredient_matrix)

def recommend_recipes(user_ingredients, num_recommendations=5):
    # Create a string representation of user's ingredients
    user_ingredients_str = ' '.join(user_ingredients)

    # Add user's ingredients to recipe data
    recipes_with_user_ingredients = recipes.copy()
    recipes_with_user_ingredients.loc[-1] = ['', 'User Recipe', user_ingredients_str]
    recipes_with_user_ingredients.reset_index(drop=True, inplace=True)

    # Vectorize ingredients
    ingredient_matrix_with_user = vectorizer.transform(recipes_with_user_ingredients['ingredients'])

    # Compute cosine similarity between user's ingredients and all recipes
    cosine_sim_with_user = cosine_similarity(ingredient_matrix_with_user, ingredient_matrix_with_user)

    # Get the index of the user recipe
    user_recipe_index = recipes_with_user_ingredients[(recipes_with_user_ingredients['recipe_name'] == 'User Recipe') & 
                                                      (recipes_with_user_ingredients['recipe_urls'] == '')].index[0]

    # Get the pairwise similarity scores for the user recipe
    similarity_scores = list(enumerate(cosine_sim_with_user[user_recipe_index]))

    # Sort the recipes based on similarity scores
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar recipes
    top_recipes = sorted_scores[1:num_recommendations+1]

    # Get the recipe names, ingredients, and URLs of the recommended recipes
    recommended_recipes = []
    for recipe in top_recipes:
        recipe_name = recipes.iloc[recipe[0]]['recipe_name']
        recipe_ingredients = recipes.iloc[recipe[0]]['ingredients']
        recipe_url = recipes.iloc[recipe[0]]['recipe_urls']
        recommended_recipes.append((recipe_name, recipe_ingredients, recipe_url))

    return recommended_recipes

emoji_feedback_mapping = {
    "😄": "Positive",
    "😐": "Neutral",
    "😞": "Negative"
}
def collect_feedback(selected_emoji):
    feedback = emoji_feedback_mapping.get(selected_emoji, "Unknown")
    feedback_df = pd.DataFrame({'Feedback': [feedback]})
    feedback_df.to_csv(r'E:\reciperecom\feedback.csv', mode='a', header=False, index=False)
# Create the Streamlit app
def main():
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')
    with col3:
        st.write(' ')

    with col2:
        st.write(' ')



    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
         f"""
        <style>
        .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
         }}
        </style>
        """,
        unsafe_allow_html=True
         )
    add_bg_from_local(r'E:\reciperecom\Background.jpg')  

    
    # Display text
    st.markdown("<h1 style='color: #A52A2A;'>          </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: #A52A2A;'>Ready to enjoy the Moroccan taste?</h1>", unsafe_allow_html=True)
    st.subheader("Hungry ! :yum::knife_fork_plate:")
    st.write("""
Moroccan cuisine is known for its aromatic spices,
bold flavors, and hearty dishes. It is a fusion of Berber,
Arab, and Mediterranean influences, which has evolved over 
centuries and reflects the country's diverse cultural heritage.
Overall, Moroccan cuisine is a vibrant and flavorful cuisine 
that reflects the country's rich history and culture. It is
 a cuisine that is well worth exploring and is sure to
   delight anyone who loves bold and complex flavors.
 """)
    st.subheader("**Given a list of ingredients, what recipes can i make:question:**")

    st.write("""
You have ingredients you don't know what to do with. 
We have more than 5000 recipes to save you :wink:. try it yourself!
""")
    # Get user input
    user_input = st.text_input("Enter ingredients (separated by commas):")
    if st.button("Recommend"):
        user_ingredients = [ingredient.strip() for ingredient in user_input.split(',')]
        recommendations = recommend_recipes(user_ingredients, num_recommendations=5)
        st.subheader("Recommended Recipes:")
        for recommendation in recommendations:
            st.write(f"- [{recommendation[0]}]({recommendation[2]})")
            st.markdown("- **Ingredients**: " + recommendation[1].replace(",", "\n  -"))
    st.subheader("Feedback:")
    selected_emoji = st.selectbox("Please select an emoji that represents your feedback:",
                                 list(emoji_feedback_mapping.keys()), format_func=lambda x: emoji.emojize(x))
    if st.button("Submit Feedback"):
        collect_feedback(selected_emoji)
        st.success("Thank you for your feedback!")    
        
if __name__ == "__main__":
    main()
