# Sistem Rekomendasi Buku Menggunakan Model Pengembangan dengan Teknik Content Based Filtering dan Collaborative Filtering

## Project Overview

Dalam era digital yang penuh informasi, literasi menghadapi tantangan menyajikan bacaan sesuai preferensi individu. Meskipun toko buku online dan layanan digital meningkatkan akses, jumlah opsi yang melimpah membuat pembaca kesulitan mencari buku yang sesuai dengan minat literasi individu. Proses pencarian buku menjadi memakan waktu dan membingungkan sehingga menghambat kegairahan membaca.

Untuk mengatasi tantangan ini, diperlukan sebuah sistem rekomendasi buku yang bisa memberikan rekomendasi personal secara akurat berdasarkan preferensi pembaca, sehingga bisa membantu mereka menemukan buku - buku yang sesuai dengan minat dan selera mereka. Selain itu, sistem rekomendasi buku juga dapat membantu memberikan dukungan dan peluang baru bagi penulis dan penerbit untuk menjangkau audiens yang lebih luas. Menurut [Khalid Anwar et al](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3356349) dalam jurnalnya berjudul _Machine Learning Techniques for Book Recommendation: An Overview_ menjelaskan bahwa sistem rekomendasi buku dapat membantu para pustakawan dalam mengelola katalog perpustakaan yang efisien dan mendukung pembaca dalam memilih buku terbaik dan sesuai selera mereka. Dari sisi penjualan menerapkan sistem rekomendasi buku dapat membantu mengelola inventaris mereka dan mendapatkan lebih banyak keuntungan. Oleh karena itu, dengan banyaknya kelebihan yang diperoleh, maka diperlukan sistem rekomendasi buku untuk meningkatkan pengalaman membaca dan mendukung perkembangan industri.

Untuk proses pengembangan model atau model development, dataset yang digunakan pada proyek ini berasal dari [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Dataset ini berisi informasi mengenai pengguna anonim beserta data demografi pengguna yang memberikan peringkat (secara eksplisit/implisit) tentang berbagai jenis buku yang dikumpulkan selama 4 minggu dari komunitas Book-Crossing pada tahun 2004. Dengan membangun sistem rekomendasi buku berdasarkan dataset tersebut, diharapkan dapat menciptakan sebuah sistem rekomendasi buku yang inovatif dan berdaya guna bagi pembaca, penulis, penerbit ataupun penjual serta berpotensi meningkatkan ekosistem industri buku secara keseluruhan.

## Business Understanding

### Problem Statements

Misalkan ada sebuah perusahaan penjual buku yang setelah sekian lama beroperasi, perusahaan tersebut berhasil mengumpulkan berbagai informasi mengenai pelanggan dan daftar buku yang berasal dari berbagai penerbit, serta terdapat rating yang diberikan oleh pelanggan untuk beberapa buku yang dibeli tersebut. Seluruh informasi ini terkumpul dalam [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data).

Seorang Data Scientist di perusahaan tersebut, ingin memanfaatkan data tersebut untuk meningkatkan transaksi penjualan di perusahaan, maka dibuatkan sebuah sistem rekomendasi buku untuk menjawab permasalahan berikut:

- Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?
- Dengan data rating yang dimiliki, bagaimana perusahaan dapat merekomendasikan buku lain yang mungkin disukai dan belum pernah dikunjungi atau dibaca oleh pengguna?

### Goals

Untuk menjawab pertanyaan tersebut, perusahaan akan membuat sebuah sistem rekomendasi dengan tujuan atau goals sebagai berikut:

- Menghasilkan sejumlah rekomendasi buku yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
- Menghasilkan sejumlah rekomendasi buku yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi atau dibaca sebelumnya dengan teknik collaborative filtering.

### Solution Statements

Berdasarkan tujuan atau goals yang sudah dijelaskan sebelumnya, maka akan dibuat sebuah sistem rekomendasi dengan alur sebagai berikut:

- **Data Understanding**, merupakan tahap awal proyek untuk memahami data yang dimiliki. Dalam kasus ini, terdapat 3 file terpisah mengenai informasi tentang users, ratings, dan books. Pada tahap ini, ada beberapa tahapan untuk memahami data antara lain:
  - **Data loading**, yaitu membaca data langsung dari dataset untuk mengetahui isi atau informasi yang ada di dalam dataset tersebut.
  - **Univariate Exploratory Data Analysis**. Pada tahap ini akan dilakukan analisis dan eksplorasi setiap variabel pada data.
  - **Data Preprocessing**. Ini merupakan tahap persiapan data sebelum data digunakan untuk proses selanjutnya. Pada tahap ini, akan dilakukan penggabungan beberapa file sehingga menjadi satu kesatuan file yang utuh dan siap digunakan dalam tahap pemodelan.
- **Data Preparation**. Pada tahap ini, data dipersiapkan dan dilakukan beberapa teknik seperti mengatasi missing value dan menyamakan nomor ISBN buku. Pada sistem rekomendasi berbasis konten (content-based filtering) yang akan dikembangkan, satu nomor ISBN mewakili satu kategori buku. Oleh karena itu, perlu pengecekan ulang dan memastikan setiap buku hanya memiliki satu nomor ISBN.
- **Modeling**. Pada proses pengembangan model akan menggunakan dua metode sebagai berikut:
  - **Model Development dengan Content Based Filtering**. Pada tahap ini, dikembangkan sistem rekomendasi dengan teknik Content Based Filtering. Teknik ini akan merekomendasikan judul buku yang sesuai dengan nama penulis buku dan disukai pengguna di masa lalu. Pada tahap ini, akan dicari representasi fitur penting dari setiap kategori buku dengan TF-IDF (Term Frequency - Inverse Document Frequency) Vertorizer dan menghitung tingkat kesamaan (similarity measure) dengan cosine similarity. Setelah itu akan menghasilkan sejumlah rekomendasi buku untuk pengguna berdasarkan kesamaan yang telah dihitung sebelumnya.
  - **Model Development dengan Collaborative Filtering**. Pada tahap ini, dikembangan sistem rekomendasi dengan teknik Collaborative Filtering. Sistem nantinya akan merekomendasikan sejumlah buku kepada pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna akan diidentifikasi nama - nama judul buku yang mirip dan belum pernah dibaca atau dibeli oleh pengguna lainnya.
- **Evaluation**, merupakan tahap untuk mengukur kinerja model dan menilai sejauh mana model berhasil dalam mencapai tujuannya dan memperoleh kesamaan fitur. Pada tahap ini digunakan metrik berupa Precision, Recall, dan F1 Score untuk model dengan content based filtering. Sedangkan, pada model dengan collaborative filtering akan menggunakan metrik Root Mean Squared Error (RMSE) sebagai metrik evaluasi model.

## Data Understanding

Data yang digunakan pada proyek ini adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data) yang diunduh dari platform Kaggle. Menurut informasi dari sumber dataset, data dikumpulkan oleh Cai-Nicolas Ziegler selama 4 minggu pada tahun 2004 dari komunitas Book-Crossing. Dataset ini berisi 278.858 pengguna (anonim tetapi dengan informasi demografis) yang memberikan 1.149.780 peringkat (secara eksplisit/implisit) tentang 271.379 buku.

Book Recommendation Dataset terdiri dari 3 file/dataset terpisah yaitu Books, Ratings, dan Users dalam format file CSV (Comma Separated Values). Dataset books terdiri 271.360 baris dan 8 kolom yang berisi informasi tentang buku antara lain nomor ISBN, judul buku, author, tahun publikasi, penerbit, dan 3 kolom untuk link URL gambar buku yang tersedia dalam 3 jenis ukuran yaitu Image-URL-S (small), Image-URL-M (medium), dan Image-URL-L (large). Dataset ratings terdiri dari 1.149.780 baris dan 3 kolom yang berisi informasi user ID atau pengguna, nomor ISBN dan peringkat buku yang berasal dari pengguna. Kemudian, dataset users yang terdiri dari 278.858 baris dan 3 kolom yang berisi informasi user ID atau pengguna, lokasi pengguna, dan usia pengguna. Berikut adalah struktur folder dari Book Recommendation Dataset yang sudah di download:

    ├── book-dataset                 <- nama folder utama.
       ├── books.csv                 <- berisi informasi buku.
       ├── ratings.csv               <- berisi informasi rating atau peringkat buku dari pengguna atau pembaca.
       └── users.csv                 <- berisi informasi pengguna atau pembaca.

### Variabel - variabel pada Book Recommendation Dataset adalah sebagai berikut:

**Variabel pada dataset Books:**

- ISBN : nomor ISBN (International Standard Book Number) dari buku.
- Book-Title : judul buku.
- Book-Author : nama penulis buku.
- Year-Of-Publication : tahun publikasi buku.
- Publisher : nama penerbit buku.
- Image-URL-S : ukuran gambar small (kecil) dari buku berupa link URL.
- Image-URL-M : ukuran gambar medium (sedang) dari buku berupa link URL.
- Image-URL-L : ukuran gambar large (besar) dari buku berupa link URL.

**Variabel pada dataset Ratings:**

- User-ID : kode unik untuk nama pengguna anonim yang memberikan penilaian.
- ISBN : nomor ISBN (International Standard Book Number) dari buku.
- Book-Rating : rating atau peringkat buku dari pengguna atau pembaca.

**Variabel pada dataset Users:**

- User-ID : kode unik untuk nama pengguna anonim.
- Location : lokasi pengguna.
- Age : usia pengguna.

### Berikut adalah beberapa tahapan untuk memahami data:

- Data Loading
- Univariate Exploratory Data Analysis
- Data Preprocessing

### Data Loading

Pada bagian ini, dataset akan dibaca secara langsung dari folder dataset yang sudah di download melalui [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data). Seperti yang telah dijelaskan sebelumnya, terdapat 3 file dataset di dalam folder yaitu Books, Ratings, dan Users yang akan digunakan untuk proses pengembangan model. Dataset dari Books, Ratings, dan Users bisa dilihat pada Tabel 1 - 3.

Tabel 1. Tampilan dari Dataset Books

|     |       ISBN | Book-Title                                                                                         | Book-Author          | Year-Of-Publication | Publisher                  | Image-URL-S                                                  | Image-URL-M                                                  | Image-URL-L                                                  |
| --: | ---------: | :------------------------------------------------------------------------------------------------- | :------------------- | ------------------: | :------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   0 | 0195153448 | Classical Mythology                                                                                | Mark P. O. Morford   |                2002 | Oxford University Press    | http://images.amazon.com/images/P/0195153448.01.THUMBZZZ.jpg | http://images.amazon.com/images/P/0195153448.01.MZZZZZZZ.jpg | http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg |
|   1 | 0002005018 | Clara Callan                                                                                       | Richard Bruce Wright |                2001 | HarperFlamingo Canada      | http://images.amazon.com/images/P/0002005018.01.THUMBZZZ.jpg | http://images.amazon.com/images/P/0002005018.01.MZZZZZZZ.jpg | http://images.amazon.com/images/P/0002005018.01.LZZZZZZZ.jpg |
|   2 | 0060973129 | Decision in Normandy                                                                               | Carlo D'Este         |                1991 | HarperPerennial            | http://images.amazon.com/images/P/0060973129.01.THUMBZZZ.jpg | http://images.amazon.com/images/P/0060973129.01.MZZZZZZZ.jpg | http://images.amazon.com/images/P/0060973129.01.LZZZZZZZ.jpg |
|   3 | 0374157065 | Flu: The Story of the Great Influenza Pandemic of 1918 and the Search for the Virus That Caused It | Gina Bari Kolata     |                1999 | Farrar Straus Giroux       | http://images.amazon.com/images/P/0374157065.01.THUMBZZZ.jpg | http://images.amazon.com/images/P/0374157065.01.MZZZZZZZ.jpg | http://images.amazon.com/images/P/0374157065.01.LZZZZZZZ.jpg |
|   4 | 0393045218 | The Mummies of Urumchi                                                                             | E. J. W. Barber      |                1999 | W. W. Norton &amp; Company | http://images.amazon.com/images/P/0393045218.01.THUMBZZZ.jpg | http://images.amazon.com/images/P/0393045218.01.MZZZZZZZ.jpg | http://images.amazon.com/images/P/0393045218.01.LZZZZZZZ.jpg |

Tabel 2. Tampilan dari dataset Ratings

|     | User-ID | ISBN       | Book-Rating |
| --: | ------: | :--------- | ----------: |
|   0 |  276725 | 034545104X |           0 |
|   1 |  276726 | 0155061224 |           5 |
|   2 |  276727 | 0446520802 |           0 |
|   3 |  276729 | 052165615X |           3 |
|   4 |  276729 | 0521795028 |           6 |

Tabel 3. Tampilan dari dataset Users

|     | User-ID | Location                           | Age |
| --: | ------: | :--------------------------------- | --: |
|   0 |       1 | nyc, new york, usa                 | nan |
|   1 |       2 | stockton, california, usa          |  18 |
|   2 |       3 | moscow, yukon territory, russia    | nan |
|   3 |       4 | porto, v.n.gaia, portugal          |  17 |
|   4 |       5 | farnborough, hants, united kingdom | nan |

Berdasarkan tampilan dataset pada Tabel 1 - 3, diperoleh informasi sebagai berikut:

- Variabel books memiliki 271.360 jenis buku dan terdiri dari 8 kolom yaitu:
  - ISBN : merupakan nomor identitas unik buku.
  - Book-Title : merupakan judul buku.
  - Book-Author : merupakan nama penulis buku.
  - Year-Of-Publication : merupakan tahun publikasi buku.
  - Publisher : merupakan nama penerbit buku.
  - Image-URL-S : merupakan link URL gambar untuk ukuran small (kecil).
  - Image-URL-M : merupakan link URL gambar untuk ukuran medium (sedang).
  - Image-URL-L : merupakan link URL gambar untuk ukuran large (besar).
- Variabel ratings memiliki 340.556 penilaian terhadap buku dan terdiri dari 3 kolom yaitu:
  - User-ID : merupakan kode unik untuk nama pengguna anonim yang memberikan penilaian.
  - ISBN : merupakan nomor identitas buku.
  - Book-Rating : merupakan penilaian yang diberikan kepada buku.
- Variabel users memiliki 278.858 nama pengguna anonim dan terdiri dari 3 kolom yaitu:
  - User-ID : merupakan kode unik untuk nama pengguna anonim.
  - Location : merupakan lokasi tempat tinggal pengguna.
  - Age : merupakan usia pengguna.

Setelah dilakukan melihat informasi dari dataset tersebut, langkah selanjutnya adalah melakukan tahapan eksplorasi pada dataset.

### Univariate Exploratory Data Analysis

Pada tahap ini, akan dilakukan analisis dan eksplorasi pada setiap variabel untuk memahami distribusi dan karakteristik individu dari variabel tersebut. Pemahaman ini nantinya akan membantu dalam menentukan pendekatan atau algoritma yang cocok diterapkan pada data. Variabel - variabel pada Book Recommendation Dataset adalah sebagai berikut:

- books : merupakan data yang berisi informasi buku.
- ratings : merupakan rating atau peringkat yang diberikan ke buku oleh pengguna atau pembaca
- users : merupakan informasi pengguna termasuk informasi demografisnya.

**Books Variabel**

Dengan menggunakan fungsi info(), diketahui bahwa dataset books yang berasal dari file books.csv memiliki 271.360 entri dan terdiri dari 8 kolom yaitu ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, dan Image-URL-L. Diketahui juga bahwa kolom 'Year-Of-Publication' bertipe data object sedangkan tahun publikasi pada umumnya memiliki tipe data integer. Oleh karena itu akan dilakukan perbaikan tipe data terlebih dahulu dengan menjalankan kode berikut: `books['Year-Of-Publication'].astype('int')`. Tetapi setelah dijalankan kode tersebut, terdapat error dengan tulisan `ValueError: invalid literal for int() with base 10: 'DK Publishing Inc'`, artinya terdapat value pada 'Year-Of-Publication' ada yang bernilai 'DK Publishing Inc'. Sepertinya ini terdapat kesalahan input, sehingga nanti akan dihapus nilai berupa teks tersebut sebelum mengubahnya ke dalam tipe data integer. Berdasarkan penelusuran, terdapat 2 nilai teks yaitu 'DK Publishing Inc' dan 'Gallimard'. Dua nilai teks ini akan dihapus dari fitur 'Year-Of-Publication'. Setelah dihapus, maka barulah dilakukan proses pengubahan tipe data pada 'Year-Of-Publication' menjadi tipe data integer.

Kemudian, langkah selanjutnya adalah menghapus variabel yang tidak diperlukan pada proses pengembangan model. Karena nantinya pada sistem rekomendasi berbasis konten (content-based filtering) akan dibuat rekomendasi berdasarkan judul buku yang sama dengan nama penulis buku yang pernah dibaca oleh pengguna. Maka informasi seperti ukuran gambar tidak diperlukan, sehingga fitur/kolom 'Image-URL-S', 'Image-URL-M', dan 'Image-URL-L' bisa dihapus. Tampilan dataset books setelah dilakukan proses penghapusan beberapa nilai dan fitur akan terlihat seperti Tabel 4.

Tabel 4. Tampilan dari Dataset Books setelah proses penghapusan beberapa nilai dan fitur

|     |       ISBN | Book-Title                                                                                         | Book-Author          | Year-Of-Publication | Publisher                  |
| --: | ---------: | :------------------------------------------------------------------------------------------------- | :------------------- | ------------------: | :------------------------- |
|   0 | 0195153448 | Classical Mythology                                                                                | Mark P. O. Morford   |                2002 | Oxford University Press    |
|   1 | 0002005018 | Clara Callan                                                                                       | Richard Bruce Wright |                2001 | HarperFlamingo Canada      |
|   2 | 0060973129 | Decision in Normandy                                                                               | Carlo D'Este         |                1991 | HarperPerennial            |
|   3 | 0374157065 | Flu: The Story of the Great Influenza Pandemic of 1918 and the Search for the Virus That Caused It | Gina Bari Kolata     |                1999 | Farrar Straus Giroux       |
|   4 | 0393045218 | The Mummies of Urumchi                                                                             | E. J. W. Barber      |                1999 | W. W. Norton &amp; Company |

Setelah melalui proses penghapusan beberapa nilai dan fitur, dataset hanya tersisa 5 kolom saja, dengan masing - masing variabel memiliki entri sebagai berikut:

- Jumlah nomor ISBN Buku: 271357
- Jumlah judul buku: 242132
- Jumlah penulis buku: 102022
- Jumlah Tahun Publikasi: 116
- Jumlah nama penerbit: 16805

Perhatikan bahwa jumlah judul buku pada dataset yaitu 242.135 sedangkan jumlah nomor ISBN buku adalah 271.357, artinya ada beberapa buku yang tidak memiliki nomor ISBN, karena satu ISBN hanya boleh dimiliki oleh satu buku saja. Untuk Kasus ini nantinya dataset akan di filter agar setiap buku dipastikan memiliki satu nomor ISBN. Selanjutnya, dilakukan distribusi data untuk melihat 10 nama penulis teratas berdasarkan jumlah buku seperti terlihat pada Gambar 1.

![cf01](https://github.com/Juwono136/book-recommendation-system-using-content-based-filtering-and-collaborative-filtering/assets/70443393/9d597829-c521-4547-9f15-bfa0bab08888)

Gambar 1. Distribusi data tentang 10 nama penulis teratas berdasarkan jumlah buku

Berdasarkan informasi pada Gambar 1, diketahui bahwa penulis dengan nama Agatha Christie menulis paling banyak buku yaitu sebanyak lebih dari 600 buku. Dari informasi ini juga diketahui jika di dalam dataset terdapat beberapa nama penulis yang menulis buku lebih dari satu judul buku.

**Ratings Variabel**

Selanjutnya, dilakukan eksplorasi pada variabel ratings, yaitu penilaian terhadap buku dari pembaca atau pengguna. Digunakan fungsi info() untuk melihat informasi dari variabel tersebut. Berdasarkan output yang diberikan, diketahui terdapat sebanyak 1.149.780 entri dan 3 kolom yaitu User-ID yang merupakan kode unik pengguna anonim yang memberikan peringkat, ISBN yang merupakan identitas berupa nomor unik buku, dan Book-Rating yang merupakan rating buku yang diberikan oleh pembaca atau pengguna. Diketahui juga terdapat 105.283 pengguna yang memberikan rating buku, jumlah buku berdasarkan ISBN yang diberikan rating adalah 340.556 buku, dan rating yang diberikan oleh masing - masing buku memiliki niliai berkisar antara 0 sampai 10, dimana 0 adalah rating paling rendah sedangkan 10 adalah rating paling tertinggi.

Seperti terlihat pada informasi sebelumnya, dataset ratings memiliki 1.149.780 baris data, dan itu merupakan jumlah yang sangat banyak. Nantinya, dataset rating ini yang akan digunakan dalam proses pengembangan model dengan collaborative filtering. Oleh karena itu, untuk menghemat alokasi memori pada saat pelatihan model nantinya, dataset rating ini tidak akan digunakan semua. Dataset rating hanya mengambil data pertama hingga data ke 5000 saja (exclude data ke 5000). Dataset ini akan digunakan untuk pengembangan model dengan collaborative filtering karena membutuhkan data rating terhadap pengguna untuk memberikan rekomendasi judul buku kepada pengguna lainnya. Untuk memudahkan supaya tidak tertukar dengan fitur lain yang serupa, variabel diubah namanya menjadi df_rating.

**Users Variabel**

Variabel terakhir yang akan dilakukan eksplorasi adalah variabel users. Variabel ini berisi informasi tentang pengguna anonim beserta demografinya. Digunakan fungsi info() untuk melihat informasi variabel. Berdasarkan output yang diberikan, diketahui terdapat 278.858 entri dan terdapat 3 variabel yaitu User-ID yang merupakan kode unik dari pengguna anonim, Location yang merupakan lokasi pengguna, dan Age yang merupakan usia pengguna. Diketahui juga terdapat beberapa pengguna yang usianya tidak diketahui. Data user berguna jika ingin membuat sistem rekomendasi berdasarkan demografi atau kondisi sosial pengguna. Namun, untuk studi kasus kali ini, tidak akan digunakan data users pada model. Pada pengembangan model, data yang digunakan adalah data books dan ratings.

### Data Preprocessing

Seperti yang sudah diketahui berdasarkan tahapan data understanding bahwa folder Book Recommendation Dataset terdiri dari 3 file terpisah yaitu books, ratings, dan users. Pada tahap ini, akan dilakukan proses penggabungan file menjadi satu kesatuan file agar sesuai dengan pengembangan model yang ingin dibuat. Variabel setelah dilakukan penggabungan menjadi 7 variabel dengan 1.149.780 baris data. Tampilan Dataset bisa dilihat pada Tabel 5, dataset inilah yang akan digunakan untuk membuat sistem rekomendasi.

Tabel 5. Tampilan dari dataset yang sudah dilakukan proses penggabungan antara variabel ratings dan books

|     | User-ID | ISBN       | Book-Rating | Book-Title                                                     | Book-Author     | Year-Of-Publication | Publisher                  |
| --: | ------: | :--------- | ----------: | :------------------------------------------------------------- | :-------------- | ------------------: | :------------------------- |
|   0 |  276725 | 034545104X |           0 | Flesh Tones: A Novel                                           | M. J. Rose      |                2002 | Ballantine Books           |
|   1 |  276726 | 0155061224 |           5 | Rites of Passage                                               | Judith Rae      |                2001 | Heinle                     |
|   2 |  276727 | 0446520802 |           0 | The Notebook                                                   | Nicholas Sparks |                1996 | Warner Books               |
|   3 |  276729 | 052165615X |           3 | Help!: Level 1                                                 | Philip Prowse   |                1999 | Cambridge University Press |
|   4 |  276729 | 0521795028 |           6 | The Amsterdam Connection : Level 4 (Cambridge English Readers) | Sue Leather     |                2001 | Cambridge University Press |

## Data Preparation

Karena terdapat dua jenis teknik yang digunakan pada proses pengembangan model, yaitu teknik content based filtering dan collaborative filtering, maka tahap persiapan data akan dibagi menjadi dua bagian, yaitu Data Preparation untuk Model Pengembangan dengan Content Based Filtering, dan Data Preparation untuk Model Pengembangan dengan Collaborative Filtering.

### Data Preparation Untuk Model Pengembangan dengan Content Based Filtering

Pada tahap ini di akan dilakukan beberapa teknik untuk mempersiapkan data seperti:

- Menghilangkan missing value.
- Menyamakan jenis buku berdasarkan ISBN.

Pada sistem rekomendasi berbasis konten (content-based filtering) yang akan dikembangkan, satu nomor ISBN mewakili satu judul buku, yang artinya nomor ISBN pada setiap buku bersifat unik. Sehingga perlu dipersiapkan terlebih dahulu datanya agar siap untuk digunakan pada proses pelatihan model.

#### Mengatasi Missing Value

Pertama dilakukan pengecekan missing value menggunakan kode berikut: `books.isnull().sum()`. Dari kode tersebut diketahui Terdapat banyak missing value pada sebagian besar fitur. Hanya fitur User-ID, ISBN, dan Book-Rating saja yang memiliki 0 missing value. Jumlah mising value terbesar ada di fitur 'Publisher' yaitu sebesar 118.650. 118.650 dari total dataset yaitu 1.149.780 merupakan jumlah yang tidak terlalu signifikan atau masih tergolong kecil. Oleh karena itu, untuk kasus ini dilakukan proses drop atau penghapusan pada missing value ini dan buatkan dalam bentuk variabel baru bernama all_books_clean. Setelah dilakukan proses penghilangan missing value, dataset terdiri dari 1.031.129 baris.

#### Menyamakan Jenis Buku Berdasarkan ISBN

Selanjutnya, sebelum masuk tahap pemodelan, diperlukan proses menyamakan judul buku berdasarkan ISBN-nya. Jika terdapat nomor ISBN yang sama pada lebih dari satu judul buku dapat menyebabkan bias pada data. Oleh karena itu harus dipastikan bahwa hanya terdapat satu nomor ISBN pada satu judul buku saja. Pada proses ini dilakukan pengecekan ulang data setelah proses cleaning pada tahap sebelumnya. Berdasarkan informasi pada dataset, diketahui bahwa jumlah nomor ISBN dengan jumlah judul buku tidak sama, artinya terdapat nomor ISBN yang sama pada lebih dari satu judul buku. Oleh karena itu, diatasi dengan mengubah datasetnya menjadi data unik sehingga nantinya siap dimasukkan ke dalam proses pemodelan dengan cara membuang data duplikat pada kolom 'ISBN'. Setelah melewati proses menyamakan jumlah dari jenis buku berdasarkan ISBN, dataset hanya tersisa 270.145 baris data. Tahap berikutnya yaitu dibuatkan dictionary untuk menentukan pasangan key-value pada data isbn_id, book_title, book_author, year_of_publication, dan publihser yang sudah disiapkan sebelumnya untuk proses pengembangan model sistem rekomendasi berbasis konten (content-based filtering). Hasil pembuatan dictionary disimpan dalam variabel bernama books_new dan terlihat seperti pada Tabel 6.

Tabel 6. Tampilan dataset books_new setelah menghilangkan missing value dan menyamakan jenis buku berdasarkan ISBN

|     | isbn       | book_title                                                     | book_author                   | year_of_publication | publisher                |
| --: | :--------- | :------------------------------------------------------------- | :---------------------------- | ------------------: | :----------------------- |
|   0 | 0000913154 | The Way Things Work: An Illustrated Encyclopedia of Technology | C. van Amerongen (translator) |                1967 | Simon &amp; Schuster     |
|   1 | 0001010565 | Mog's Christmas                                                | Judith Kerr                   |                1992 | Collins                  |
|   2 | 0001046438 | Liar                                                           | Stephen Fry                   |                   0 | Harpercollins Uk         |
|   3 | 0001046713 | Twopence to Cross the Mersey                                   | Helen Forrester               |                1992 | HarperCollins Publishers |
|   4 | 000104687X | T.S. Eliot Reading \The Wasteland\" and Other Poems"           | T.S. Eliot                    |                1993 | HarperCollins Publishers |

Berdasarkan informasi sebelumnya, dataset memiliki sekitar 270.145 baris data. Karena dataset tersebut terlalu banyak dan secara otomatis alokasi memori yang digunakan nantinya akan sangat banyak untuk memproses seluruh data pada pengembangan model, maka pada proyek ini hanya akan mengambil data pertama hingga data ke 20.000 (exlude data ke 20.000). Dataset books_new inilah yang akan digunakan pada proses pengembangan model dengan teknik Content Based Filtering.

### Data Preparation Untuk Model Pengembangan dengan Collaborative Filtering

Pada model pengembangan dengan collaborative filtering, data nantinya akan dibagi menjadi data training dan data validasi dalam proses pelatihan model, sebelum di bagi menjadi data training dan data validasi, data harus dipersiapkan terlebih dahulu. Data rating harus diubah ke dalam bentuk matriks numerik agar nantinya mempermudah proses pelatihan model sehingga model menjadi mudah mengenali/mempelajari data tersebut. Sebelum itu dilakukan, Pada tahap ini di dilakukan beberapa teknik untuk mempersiapkan data seperti menyandikan (encode) fitur 'User-ID' dan 'ISBN' ke dalam indeks integer, memetakan 'User-ID' dan 'ISBN'ke dataframe yang berkaitan, dan yang terakhir mengecek beberapa hal dalam data seperti jumlah pengguna, jumlah buku, kemudian mengubah nilai rating menjadi float agar bisa digunakan pada proses pelatihan model. Setelah tahap persiapan data sudah selesai, data siap dibagi menjadi data training dan data validasi untuk proses training model.

## Modeling

### Model Development dengan Content Based Filtering

Pada tahap ini, dikembangkan model dengan teknik Content Based Filtering. Content Based Filtering adalah salah satu pendekatan dalam sistem rekomendasi yang menggunakan informasi atau "konten" dari item atau pengguna untuk membuat rekomendasi. Ide dasarnya adalah mencocokkan preferensi pengguna dengan karakteristik atau konten dari item yang telah dilihat atau disukai oleh pengguna sebelumnya. Misalkan, jika seorang pengguna menyukai atau pernah membeli buku dengan judul "Introduction to Machine Learning" dan buku tersebut memiliki fitur berupa nama penulis buku yaitu "Alex Smola", maka sistem akan mencari buku lain dengan fitur serupa dan merekomendasikannya dalam bentuk top-N recommendation kepada pengguna tersebut.

Kelebihan teknik Content Based Filtering:

- Memberikan rekomendasi yang personal untuk setiap pengguna berdasarkan preferensi unik pengguna.
- Cocok untuk mengatasi masalah cold-start (ketika sedikit data pengguna tersedia) dan dapat memberikan rekomendasi yang baik sejak awal.
- Tidak tergantung pada data pengguna lain.
- Cocok untuk item dengan fitur atau atribut yang mudah diukur atau diidentifikasi, seperti genre, penulis, atau karakteristik lainnya.

Kekurangan teknik Content Based Filtering:

- Tidak efektif dalam menangani kejutan, karena hanya merekomendasikan item yang serupa dengan yang sudah diketahui pengguna.
- Kesulitan dalam menangkap preferensi yang kompleks atau dinamis dari pengguna, terutama jika item yang serupa tidak mencerminkan preferensi yang mendalam.
- Ada risiko "filter bubble" di mana pengguna hanya menerima rekomendasi yang sesuai dengan preferensi pengguna yang sudah diketahui, tanpa diverifikasi.

Pada proses pengembangan model dilakukan pencarian representasi fitur penting dari setiap judul buku dengan TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer. TF-IDF vectorizer adalah alat yang digunakan untuk mengonversi dokumen teks menjadi representasi vektor berdasarkan nilai TF-ID setiap kata dalam dokumen tersebut. TF (Term Frequency) mengukur seberapa sering suatu kata muncul dalam suatu dokumen. Sedangkan, IDF mengukur seberapa unik atau jarang suatu kata muncul dalam seluruh koleksi dokumen. Vektor ini nanti digunakan untuk melakukan proses pencarian representasi fitur penting dari setiap judul buku berdasarkan nama penulis buku pada model yang dikembangkan dengan teknik Content Based Filtering. Pada proyek ini digunakan fungsi [tfidfvectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) dari library Sklearn. Pertama, dilakukan import fungsi tfidfvectorizer() untuk melakukan perhitungan idf pada data book_author, kemudian dilanjutkan dengan melakukan mapping array dari fitur index integer ke fitur nama menggunakan fungsi get_feature_name_out(). Langkah selanjutnya yaitu melakukan fit dan transformer data ke dalam bentuk matriks dengan ukuran (20000, 8746) agar bisa dikenali saat proses mencari kesamaan fitur. Nilai 20000 merupakan ukuran data dan 8746 merupakan matriks nama penulis buku. Untuk menghasilkan vektor tf-idf dalam bentuk matriks, digunakan fungsi todense(). Terakhir, tampilkan matriks tf-idf untuk beberapa judul buku dan nama penulis buku dalam bentuk dataframe, dimana kolom diisi dengan nama penulis buku sedangkan baris diisi dengan judul buku seperti terlihat pada Tabel 7.

Tabel 7. Dataframe dari matriks tf-idf

| book_title                                                                    | maisie | wallace | muir | civil | wole | keane | peris | gregorian | latour | brandt | suu | georgene | ringer | dainty | julie |
| :---------------------------------------------------------------------------- | -----: | ------: | ---: | ----: | ---: | ----: | ----: | --------: | -----: | -----: | --: | -------: | -----: | -----: | ----: |
| The Korean Cinderella                                                         |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| Best Girl                                                                     |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| The Complete Idiot's Guide to Adoption (Complete Idiot's Guides)              |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| Yo, Poe                                                                       |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| Cool Shade                                                                    |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| Dance Hall of the Dead (Joe Leaphorn Novels)                                  |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| Deadly Legacy                                                                 |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| The New Individualists: The Generation After the Organization Man             |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| The Geometry of Love: Space, Time, Mystery, and Meaning in an Ordinary Church |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |
| The Woman Who Was Not All There                                               |      0 |       0 |    0 |     0 |    0 |     0 |     0 |         0 |      0 |      0 |   0 |        0 |      0 |      0 |     0 |

Berdasarkan Tabel 7, matriks tf-idf berhasil mengidentifikasi representasi fitur penting dari setiap kategori judul buku dengan fungsi tfidfvectorizer. Pada Kasus ini dataset hanya ditampilkan berupa sampel data sehingga tidak terlihat keseluruhan matriks. Dari 20000 data hanya dipilih sampel data acak yang terdiri dari 10 judul buku pada baris vertikal dan 15 nama penulis buku pada baris horizontal.

Sementara itu, untuk menghitung derajat kesamaan (similarity degree) antar judul buku digunakan teknik cosine similarity. Metode ini digunakan untuk mengukur sejauh mana kesamaan antar dua vektor dalam ruang berdimensi banyak. Cosine similarity mengukur sudut kosinus antara dua vektor, dan semakin kecil sudutnya, semakin besar kesamaan antara vektor - vektor tersebut. Pada proyek ini digunakan fungsi [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari library Sklearn. Pertama, digunakan fungsi cosine_similarity() dari library sklearn untuk menghitung nilai cosine similarity pada matriks tf-idf yang sudah dibuat sebelumnya. Dengan fungsi cosine_similarity(), didapat nilai kesamaan (similarity) antar judul buku berupa matriks kesamaan dalam bentuk array. Kemudian, dibuatkan dataframe dari hasil perhitungan cosine similarity dengan baris dan kolom berupa nama judul buku. Tampilan dataframe hasil perhitungan cosine similarity terlihat pada Tabel 8.

Tabel 8. Dataframe hasil perhitungan cosine similarity

| book_title                                                                                                   | Food--Your Miracle Medicine | Youth, Heart of Darkness, the End of the Tether (Penguin Great Books of the 20th Century) | Our Religions : The Seven World Religions Introduced by Preeminent Scholars from Each Tradition | The Second Man | Ask for May, Settle for June (A Doonesbury book) |
| :----------------------------------------------------------------------------------------------------------- | --------------------------: | ----------------------------------------------------------------------------------------: | ----------------------------------------------------------------------------------------------: | -------------: | -----------------------------------------------: |
| Eat More, Weigh Less: Dr. Dean Ornish's Life Choice Program for Losing Weight Safely While Eating Abundantly |                           0 |                                                                                         0 |                                                                                               0 |              0 |                                                0 |
| Moving Molly                                                                                                 |                           0 |                                                                                         0 |                                                                                               0 |              0 |                                                0 |
| How to be a very important person (McGraw-Hill paperbacks)                                                   |                           0 |                                                                                         0 |                                                                                               0 |              0 |                                                0 |
| Jade Tiger                                                                                                   |                           0 |                                                                                         0 |                                                                                               0 |              0 |                                                0 |
| Development of the Nervous System                                                                            |                           0 |                                                                                         0 |                                                                                               0 |              0 |                                                0 |

Berdasarkan Tabel 8, dengan cosine similarity, berhasil diidentifikasi kesamaan antara satu judul buku dengan judul buku lainnya. Diketahui Shape (20000, 20000) yang merupakan ukuran matriks similarity dari data. Berdasarkan data yang ada, matriks diatas pada Tabel 8 sebenarnya berukuran 20000 judul buku x 20000 judul buku (masing - masing dalam sumbu X dan Y). Artinya, telah berhasil mengidentifikasi tingkat kesamaan pada 20000 judul buku. Tapi disini tidak ditampilkan semua datanya karena keterbatasan alokasi memori pada perangkat. Oleh karena itu, hanya dipilih 5 judul buku pada baris vertikal dan 5 judul buku pada baris horizontal. Dengan data kesamaan (similarity) judul buku yang diperoleh sebelumnya, akan dilakukan proses rekomendasi daftar judul buku yang mirip dengan judul buku yang sebelumnya pernah dibeli atau dibaca oleh pengguna.

#### Mendapatkan Rekomendasi

Pada tahap ini, dibuatkan sebuah fungsi bernama book_recommendations dengan beberapa parameter sebagai berikut:

- book_title : nama judul buku (index kemiripan dataframe).
- similarity_data : Dataframe mengenai similarity yang telah didefinisikan sebelumnya.
- items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah 'book_title' dan 'book_author'.
- k : Jumlah top-N recommendation yang diberikan oleh sistem rekomendasi. Secara default, k bernilai 5.

fungsi book_recommendation adalah fungsi yang dibuat untuk menampilkan hasil rekomendasi berbasis konten berupa judul buku dengan nama penulis yang sama dengan judul buku yang pernah dibeli atau dibaca oleh pengguna. Sebelum fungsi dibuat, perlu diingat bahwa definisi dari sistem rekomendasi yang menyatakan bahwa keluaran sistem adalah berupa top-N recommendation. Oleh karena itu, nantinya untuk mendapatkan rekomendasi perlu diberikan sejumlah rekomendasi judul buku pada pengguna yang diatur pada parameter k.

Pada fungsi book_recommendations digunakan juga fungsi argpartition() untuk proses pengambilan sejumlah nilai k tertinggi dari similarity data. Selanjutnya, mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Data ini dimasukkan ke dalam variabel bernama 'closest'. Berikutnya, perlu dihapus book_title yang dicari agar tidak muncul dalam daftar rekomendasi. Dalam kasus ini, akan dicari judul buku yang mirip dengan judul buku yang nanti di input dalam book_title, sehingga perlu drop book_title agar tidak muncul dalam daftar rekomendasi yang diberikan nanti. Contoh judul buku yang digunakan terlihat pada Tabel 9.

Tabel 9. Contoh judul buku yang digunakan untuk menampilkan rekomendasi

|      |       isbn | book_title                                                                                | book_author   | year_of_publication | publisher          |
| ---: | ---------: | :---------------------------------------------------------------------------------------- | :------------ | ------------------: | :----------------- |
| 6448 | 0060654775 | Entering the Silence : Becoming a Monk and a Writer (The Journals of Thomas Merton, V. 2) | Thomas Merton |                1997 | HarperSanFrancisco |

Pada Tabel 9, terlihat bahwa judul buku 'Entering the Silence : Becoming a Monk and a Writer (The Journals of Thomas Merton, V. 2)' yang ditulis oleh 'Thomas Merton'. Selanjutnya, digunakan fungsi book_recommendations untuk menampilkan rekomendasi 5 buku teratas yang direkomendasikan sistem berdasarkan judul buku di Tabel 9. Hasilnya terlihat pada Tabel 10.

Tabel 10. Hasil rekomendasi 5 judul buku teratas dengan kateogri nama penulis (book_author)

|     | book_title                                                                                                                           | book_author   |
| --: | :----------------------------------------------------------------------------------------------------------------------------------- | :------------ |
|   0 | Dancing in the Water of Life: Seeking Peace in the Hermitage (Merton, Thomas//Journal of Thomas Merton)                              | Thomas Merton |
|   1 | Dialogues with Silence: Prayers and Drawings                                                                                         | Thomas Merton |
|   2 | The Other Side of the Mountain: The End of the Journey (Merton, Thomas//Journal of Thomas Merton)                                    | Thomas Merton |
|   3 | Run to the Mountain : The Story of a VocationThe Journal of Thomas Merton, Volume 1: 1939-1941 (The Journals of Thomas Merton, V. 1) | Thomas Merton |
|   4 | Turning Toward the World: The Pivotal Years (The Journals of Thomas Merton, Volume 4: 1960-1963)                                     | Thomas Merton |

Berdasarkan Tabel 10, dapat dilihat bahwa sistem berhasil merekomendasikan 5 judul buku teratas dengan kategori nama penulis (book_author) yaitu 'Thomas Merton'.

### Model Development dengan Collaborative Filtering

Pada proses pengembangan model kali ini, akan diterapkan teknik collaborative filtering untuk membuat sistem rekomendasi. Teknik ini membutuhkan data rating dari pengguna atau pembaca. Collaborative filtering adalah salah satu metode dalam sistem rekomendasi yang memprediksi preferensi atau minat pengguna terhadap item berdasarkan informasi dari pengguna lain (kolaborasi). Ide dasar dibalik collaborative filtering adalah bahwa pengguna yang memiliki preferensi serupa dalam masa lalu cenderung memiliki preferensi serupa untuk item di masa depan. Pada proyek ini akan dibuat model collaborative filtering berdasarkan kesamaan antar pengguna (User-Based Collaborative Filtering).

Kelebihan Collaborative Filtering:

- Dapat memberikan rekomendasi yang sangat personal kepada pengguna karena memanfaatkan preferensi dan perilaku pengguna secara langsung.
- Tidak memerlukan pengetahuan mendalam tentang item atau produk yang direkomendasikan. Collaborative filtering hanya memerlukan pola preferensi pengguna.
- Dapat menangani item baru yang belum memiliki sejarah penggunaan atau peringkat, karena rekomendasi didasarkan pada pola perilaku keseluruhan pengguna.

Kekurangan Collaborative Filtering:

- Collaborative filtering mengalami kesulitan saat menghadapi masalah cold start, karena tanpa data sejarah, sulit memberikan rekomendasi yang akurat.
- Menghadapi masalah dengan data yang bersifat sparse (jarang), di mana mayoritas pengguna hanya memberikan peringkat atau preferensi untuk sebagian kecil item.
- Perfoma collaborative filtering dapat menurun seiring dengan pertambahan jumlah pengguna atau item, karena perhitungan kesamaan antar pengguna atau item dapat menjadi lebih kompleks dan membutuhkan sumber daya yang signifikan.

Pengembangan model dengan Collaborative filtering pada proyek ini akan menghasilkan rekomendasi sejumlah judul buku yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna, akan diidentifikasi nama - nama judul buku yang mirip dan belum pernah dibaca atau dibeli oleh pengguna untuk direkomendasikan.

Setelah tahapan persiapan data untuk pengembangan model ini sudah dilakukan di bagian data preparation sebelumnya, langkah selanjutnya dilakukan pembagian data untuk data training dan data validasi, kemudian dilanjutkan dengan proses training model. Pada proses training, model menghitung skor kecocokan antara pengguna dan judul buku dengan teknik embedding. Pertama, dilakukan proses embedding terhadap data pengguna dan judul buku. Selanjutnya, lakukan operasi perkalian dot product antara embedding pengguna dan judul buku. Selain itu, ditambahkan bias untuk setiap pengguna dan judul buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Model dibuatkan class RecommenderNet dengan [keras Model class](https://keras.io/api/models/model/). Kode class RecommenderNet ini terinspirasi dari tutorial dalam situs [keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) dengan beberapa adaptasi layer yang menyesuaikan dengan kasus yang sedang dikerjakan. Model akan menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan Root Mean Squared Error (RMSE) sebagai metrik evaluasi.

#### Membagi Data Untuk Training dan Validasi

Sebelum dilakukan proses pembagian data. Dataset akan diacak terlebih dahulu agar distribusinya menjadi random. Setelah itu, dilakukan proses pembagian data menjadi data train dan validasi dengan komposisi 90:10. Namun sebelumnya, perlu dipetakan (mapping) data user dan judul buku menjadi satu value terlebih dahulu. Kemudian, dibuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training.

#### Proses Training

Pada proses training model, model akan menghitung skor kecocokan antara pengguna dan judul buku dengan teknik embedding. Pertama, dilakukan proses embedding terhadap data user dan book_title. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan book_title. Selain itu, juga dapat ditambahkan bias untuk setiap user dan book_title. Skor kecocokan ditetapkan dalam skala [0, 1] dengan fungsi aktivasi sigmoid.

Di sini, Model dibuatkan class RecommenderNet dengan [keras Model class](https://keras.io/api/models/model/). Kode class RecommenderNet ini terinspirasi dari tutorial dalam situs [keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) dengan beberapa adaptasi layer yang menyesuaikan dengan kasus yang sedang dikerjakan. Class RecommenderNet ini akan berisi layer yang akan melatih model. Setelah layer model sudah dibuat, dilakukan proses compile terhadap model menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation.

Berdasarkan hasil proses training model, didapat hasil yang cukup memuaskan dan model konvergen pada epochs sekitar 50. Dari proses ini, diperoleh nilai Root Mean Squared Error (RMSE) sebesar sekitar 0.2939 dan RMSE pada data validasi sebesar 0.3353. Nilai ini cukup bagus untuk sistem rekomendasi. Untuk mengetahui hasil dari pengembangan model, langkah selanjutnya adalah mendapatkan rekomendasi judul buku berdasarkan model yang dikembangan.

#### Mendapatkan Rekomendasi Judul Buku

Untuk mendapatkan rekomendasi judul buku, pertama diambil sampel user secara acak dan definisikan variabel `book_not_readed` yang merupakan daftar buku yang belum pernah dibaca atau dibeli oleh pengguna. variabel `book_not_readed` inilah yang akan menjadi judul buku yang direkomendasikan oleh sistem. Variabel `book_bot_visited` diperoleh dengan menggunakan operator bitwise (~) pada variabel `book_readed_by_user`. Selanjutnya, untuk memperoleh rekomendasi judul buku, digunakan fungsi model.predict() dari library Keras dan hasil output akan menampilkan top-N recommendation berdasarkan preferensi pengguna seperti terlihat pada Tabel 11 dan Tabel 12.

Tabel 11. Hasil top-N recommendation berupa buku dengan nilai rating tertinggi dari user

|     | Book Title                | Book Author     |
| --: | :------------------------ | :-------------- |
|   0 | My \Star Trek\" Memories" | William Shatner |

Tabel 12. Hasil top-N recommendation berupa 10 buku teratas yang direkomendasikan

|     | Book Title                                                                                    | Book Author         |
| --: | :-------------------------------------------------------------------------------------------- | :------------------ |
|   0 | Silence of the Sirens                                                                         | Adel Garcia Morales |
|   1 | Politically Correct Bedtime Stories: Modern Tales for Our Life and Times                      | James Finn Garner   |
|   2 | On Writing Well, 25th Anniversary : The Classic Guide to Writing Nonfiction (On Writing Well) | William Zinsser     |
|   3 | Get Shorty                                                                                    | Elmore Leonard      |
|   4 | One for the Money (Stephanie Plum Novels (Paperback))                                         | Janet Evanovich     |
|   5 | Junky                                                                                         | William Burroughs   |
|   6 | The Oranging of America and Other Stories                                                     | Max Apple           |
|   7 | Joke                                                                                          | Milan Kundera       |
|   8 | Tales of Mystery and Terror (Puffin Classics)                                                 | Edgar Allan Poe     |
|   9 | War and Peace (Penguin Classics)                                                              | Leo Tolstoy         |

Berdasarkan Tabel 11 dan Tabel 12, model telah berhasil membuat rekomendasi kepada user. Hasil tersebut adalah rekomendasi untuk user dengan id 276798. Dari output tersebut, dapat dibandingkan antara 'Book with high ratings from user' dan 'Top 10 books recomendation' untuk user. Perhatikan, beberapa judul buku rekomendasi menyediakan nama penulis bukunya juga yang sesuai dengan rating user. Diperoleh 10 rekomendasi teratas buku yang disertai juga dengan nama penulisnya untuk user tersebut serta terdapat 1 judul buku yang merupakan buku dengan rating tertinggi dari user.

## Evaluation

### Evaluasi Model dengan Content Based Filtering

Metrik yang digunakan untuk evaluasi model dengan content based filtering di proyek kali ini adalah Precision, Recall, dan F1-Score. Metrik ini adalah metrik yang umum digunakan untuk mengukur kinerja model. Precision merupakan rasio item yang revelan yang dihasilkan oleh model terhadap total item yang dihasilkan. Recall merupakan rasio item relevan yang dihasilkan oleh model terhadap total item yang seharusnya direkomendasikan. Sedangkan, F1 Score adalah gabungan dari Precision dan Recall, memberikan nilai tunggal yang mengukur keseimbangan antara keduanya. Berikut adalah rumus untuk menghitung Precision, Recall, dan F1 Score pada model sistem rekomendasi berbasis konten:

$$Precision = \frac{Jumlah\ item\ revelan\ yang\ dihasilkan}{Total\ item\ yang\ dihasilkan}$$

$$Recall = \frac{Jumlah\ item\ relevan\ yang\ dihasilkan}{Total\ item\ yang\ seharusnya\ direkomendasikan}$$

$$F1\ Score = 2 * \frac{Precision * Recall}{Precision + Recall}$$

Sebelum menghitung nilai evaluasi metrik menggunakan precision, recall dan f1 score, diperlukan sebuah data yang terdiri dari label sebenarnya dan digunakan untuk menilai hasil prediksi model, data ini disebut sebagai data **ground truth**. Data ground truth pada proyek ini dibuat menggunakan hasil derajat kesamaan yang dihitung menggunakan teknik cosine similarity, dimana setiap baris dan kolom mewakili judul buku, dan nilai di setiap sel pada dataframe mewakili label. Angka 1 untuk similar, dan angka 0 untuk tidak similar. Perlu ditetapkan juga sebuah nilai ambang batas atau threshold untuk memutuskan apakah nilai similarity antara dua item harus dianggap 1 (similar) atau 0 (tidak similar). Nilai ambang batas atau threshold ditetapkan sebesar 0.5 pada proyek ini. Nilai threshold ini disesuaikan dengan kebutuhan dan karakteristik setelah melihat hasil rekomendasi sebelumnya. Lalu dibuatkan matriks ground truth menggunakan fungsi np.where() dari NumPy. Matriks ini akan memiliki nilai 1 di posisi di mana nilai cosine similarity antara dua item lebih besar atau sama dengan nilai threshold yang ditetapkan, dan nilai 0 di posisi di mana nilai similarity di bawah threshold. Kemudian, setelah matriks dibuat, hasilnya disajikan dalam bentuk dataframe. Baris dan kolom Dataframe ground truth ini diindeks menggunakan judul buku dari data.

Setelah dibuatkan matriks ground truth yang berisi label sebenarnya dari hasil cosine similarity. Selanjutnya, dilakukan proses perhitungan evaluasi model dengan metrik precision, recall, dan f1 score. Pertama, mengimport fungsi `precision_recall_fscore_support` dari [library Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html) yang digunakan untuk menghitung precision, recall dan f1 score. Lalu karena keterbatasan alokasi memori pada perangkat, data hanya diambil sekitar 10000 sampel dari cosine similarity dan ground truth matriks. Hal ini dilakukan untuk mempercepat proses perhitungan, terutama karena ukuran matriks yang cukup besar. Kemudian, matriks cosine similarity dan ground truth dikonversi menjadi array satu dimensi agar mempermudah perbandingan dan perhitungan metrik evaluasi.

Hasilnya disimpan dalam array `predictions`. Terakhir, digunakan fungsi `precision_recall_fscore_support` untuk menghitung precision, recall, dan f1 score. Parameter average='binary' digunakan karena sedang mengukur kinerja dalam konteks klasifikasi biner (1 atau 0). Parameter 'zero_division=1' digunakan untuk menghindari pembagian dengan nol jika ada kelas yang tidak terdapat di prediksi. Hasil evaluasi metriks didapat adalah sebagai berikut:

- Precision: 1.0
- Recall: 1.0
- F1-score: 1.0

Berdasarkan hasil evaluasi, didapat nilai dari masing - masing metrik evaluasi yaitu precision, recall dan F1 Score. Nilai Precision didapat sebesar 1.0, artinya semua prediksi positif model adalah benar dan tidak terdapat false positive. Nilai recall didapat nilai 1.0 menunjukkan bahwa model berhasil mengidentifikasi sekitar 100% dari semua item yang sebenarnya relevan. Nilai F1 Score didapat sekitar 1.0 juga, ini menunjukkan keseimbangan yang baik antara precision dan recall dan model cenderung memberikan hasil yang sangat baik untuk kedua kelas (positif dan negatif). Kesimpulannya, berdasarkan hasil metrik evaluasi tersebut model bekerja dengan sangat baik dalam memberikan rekomendasi item dengan content based filtering.

### Evaluasi Model dengan Collaborative Filtering

Seperti yang sudah dilihat pada proses pelatihan model di bagian modeling. Metrik yang digunakan untuk melakukan evaluasi model pada model dengan Collaborative Filtering di proyek ini adalah [Root Mean Squared Error (RMSE)](https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/). RMSE adalah metrik evaluasi yang umum digunakan untuk mengukur seberapa baik model memprediksi nilai kontinu dengan membandingkan nilai prediksi dengan nilai sebenarnya. Dalam konteks collaborative filtering, RMSE biasanya digunakan untuk mengukur seberapa baik model kolaboratif dalam memprediksi preferensi pengguna terhadap item. RMSE didefinisikan dalam persamaan berikut:

$$RMSE = \sqrt{\frac{1}{N} \Sigma_{i=1}^N({y_i}- y\_pred_i)^2}$$

Keterangan:

- N adalah jumlah prediksi.
- yi adalah nilai sebenarnya dari preferensi pengguna terhadap item.
- y_pred adalah prediksi model terhadap preferensi pengguna terhadap item.

Berdasarkan hasil proses training model pada tahap modeling, diperoleh hasil pelatihan berupa informasi RMSE di data train dan validasi. Untuk melihat visualisai proses training model, dilakukan proses plot metrik evaluasi dengan matplotlib dan terlihat seperti Gambar 2.

![rmse_06](https://github.com/Juwono136/book-recommendation-system-using-content-based-filtering-and-collaborative-filtering/assets/70443393/b2e1cc94-4931-4e11-b21b-25e13e92580e)

Gambar 2. Visualisasi dari metrik evaluasi model

Berdasarkan Gambar 2 didapat hasil visualisasi metrik evaluasi RMSE terhadap model yang dikembangkan, terlihat hasil model konvergen pada epochs sekitar 50 dan berdasarkan plot metriks model terlihat memberikan nilai MSE yang cukup kecil. Dari proses ini, diperoleh nilai error akhir sebesar 0.2939 dan error pada data validasi sebesar 0.3353. Nilai tersebut menunjukkan hasil yang cukup baik untuk sistem rekomendasi yang dihasilkan. Semakin kecil nilai RMSE, semakin baik model dalam memprediksi preferensi pengguna terhadap item. Hal inilah yang menyebabkan hasil rekomendasi dari model cukup akurat.
