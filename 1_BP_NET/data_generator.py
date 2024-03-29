import csv
import random

def generate_csv(filename, num_rows, composer_weights=None, w1=None,w2=None):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV表头
        writer.writerow(["Composer", "Genre1", "Genre2" ,"Liked"])
        i=0
        while i<num_rows:
            composer = random.choices(list(composer_weights.keys()), weights=list(composer_weights.values()))[0]
            genre1 = random.choices(list(genre_weights1.keys()), weights=list(genre_weights1.values()))[0]
            genre2 = random.choices(list(genre_weights2.keys()), weights=list(genre_weights2.values()))[0]
            
            tw=composer_weights[composer]+genre_weights1[genre1]+genre_weights2[genre2]
            if tw<=1.5:
                liked=0
            else:
                liked=1
            
            if not (genre1=="none" and genre2=="none"):
                writer.writerow([composer, genre1, genre2, liked])
                i+=1
if __name__ == "__main__":
    # 权重字典
    composer_weights = {"taylor":0.9, "jay":0.8, "lucy":0.7, "troye":0.5, "beyonce":0.3, "kanye":0.2, "micheal":0.1}
    genre_weights1 = {"pop":0.9, "rock":0.7, "none":0.5, "electronic":0.3, "jazz":0.1}
    genre_weights2 ={"absolute":0.9, "rap":0.7, "none":0.5, "world music":0.3, "classical":0.1}
    generate_csv("music_data.csv", num_rows=2000, composer_weights=composer_weights, w1=genre_weights1, w2=genre_weights2)
