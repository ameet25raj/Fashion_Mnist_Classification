# Fashion_Mnist_Classification_Project (Deep learning on CNN Architecture)

👗👖👠 Dive into the realm of fashion with the captivating 𝐃𝐞𝐞𝐩 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐅𝐚𝐬𝐡𝐢𝐨𝐧 𝐌𝐍𝐈𝐒𝐓 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐜𝐚𝐭𝐢𝐨𝐧 𝐩𝐫𝐨𝐣𝐞𝐜𝐭!
Leveraging the 𝐅𝐚𝐬𝐡𝐢𝐨𝐧 𝐌𝐍𝐈𝐒𝐓 dataset—a fashion-centric adaptation of the iconic 𝐌𝐍𝐈𝐒𝐓 dataset—embarking on an exhilarating journey to classify fashion  items. With 𝟔𝟎,𝟎𝟎𝟎 training examples and 𝟏𝟎,𝟎𝟎𝟎 test samples, each image is a fixed 𝟐𝟖𝐱𝟐𝟖 pixels. Utilize cutting-edge 𝐝𝐞𝐞𝐩 𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 techniques to accurately categorize garments like coats, trousers, pullovers, shoes, and more.  

👠👕🧢 Welcome to the 𝐅𝐚𝐬𝐡𝐢𝐨𝐧 𝐌𝐍𝐈𝐒𝐓 Classifier project! In this project, have to build  a 𝐝𝐞𝐞𝐩 𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐦𝐨𝐝𝐞𝐥 to classify fashion items from the 𝐅𝐚𝐬𝐡𝐢𝐨𝐧 𝐌𝐍𝐈𝐒𝐓 dataset.

Let's embark on this exciting journey of fashion classification! 🛍️👠👖🕶️🧥🧢👗👟👕👜


 ![Screenshot 2024-09-16 223019](https://github.com/user-attachments/assets/cd356ff1-59af-4443-9653-f58e8e1d925b)

𝐏𝐫𝐨𝐣𝐞𝐜𝐭 𝐨𝐯𝐞𝐫𝐯𝐢𝐞𝐰💫👇

1. 𝐈𝐦𝐩𝐨𝐫𝐭 𝐋𝐢𝐛𝐫𝐚𝐫𝐢𝐞𝐬:  Importing  TensorFlow, Keras, NumPy, Pandas, Matplotlib,  Seaborn etc for data handling, visualization, and model building.

2. 𝐋𝐨𝐚𝐝 𝐃𝐚𝐭𝐚𝐬𝐞𝐭: Utilizing TensorFlow or Keras, load the Fashion MNIST dataset containing 60,000 training images and 10,000 testing images, each with a resolution of 28x28 pixels.

3. 𝐂𝐡𝐞𝐜𝐤𝐢𝐧𝐠 𝐌𝐢𝐬𝐬𝐢𝐧𝐠 𝐕𝐚𝐥𝐮𝐞𝐬: Exploratory data analysis will help to ensure the data is clean and ready for modeling.

4. 𝐕𝐢𝐬𝐮𝐚𝐥𝐢𝐳𝐢𝐧𝐠 𝐈𝐦𝐚𝐠𝐞𝐬: Matplotlib & seaborn will aid us in visualizing sample images from the dataset, providing insight into different fashion items.

5. 𝐂𝐡𝐚𝐧𝐠𝐢𝐧𝐠 𝐃𝐢𝐦𝐞𝐧𝐬𝐢𝐨𝐧𝐬: NumPy will help adjust input data dimensions to fit our model's requirements, such as converting images to a 4D array.

6. 𝐅𝐞𝐚𝐭𝐮𝐫𝐞 𝐒𝐜𝐚𝐥𝐢𝐧𝐠: Scaling  pixel values to a range of 0 to 1, ensuring effective model learning without bias from varying pixel intensities.

7. 𝐁𝐮𝐢𝐥𝐝𝐢𝐧𝐠 𝐭𝐡𝐞 𝐌𝐨𝐝𝐞𝐥: Constructing a CNN architecture with convolutional layers, pooling layers, and dense layers using Keras.

8. 𝐓𝐫𝐚𝐢𝐧𝐢𝐧𝐠 𝐭𝐡𝐞 𝐌𝐨𝐝𝐞𝐥: Model training on the training dataset with validation data to adjust parameters based on observed performance.

9. 𝐓𝐞𝐬𝐭𝐢𝐧𝐠 𝐚𝐧𝐝 𝐄𝐯𝐚𝐥𝐮𝐚𝐭𝐢𝐨𝐧: Assessing model performance on the test dataset using metrics like accuracy, precision, recall, and F1-score.

10. 𝐂𝐨𝐧𝐟𝐮𝐬𝐢𝐨𝐧 𝐌𝐚𝐭𝐫𝐢𝐱: Visualizing a confusion matrix to gain insights into model performance and error types.

11. 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐜𝐚𝐭𝐢𝐨𝐧 𝐑𝐞𝐩𝐨𝐫𝐭: Generating a detailed summary of model performance across different classes, including precision, recall, and F1-score.


Throughout the project,  providing  relevant visualizations and code snippets for a comprehensive understanding. 

🌟 "𝐒𝐞𝐞𝐤𝐢𝐧𝐠 𝐂𝐨𝐥𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐢𝐨𝐧 𝐚𝐧𝐝 𝐃𝐢𝐬𝐜𝐮𝐬𝐬𝐢𝐨𝐧" :-

I'm passionate about data science.
Feel free to reach out and let's connect!




