import streamlit as st
import random
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import zipfile
from streamlit_option_menu import option_menu

#function 
def gen_prime():
    while True:
        num = random.randint(2**7, 2**10)
        if is_prime(num):
            return num
        
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def read_image(image_path):
  image = img.imread(image_path)
  return image

def image_shape(image):
  w, h = image.shape[:2]
  return w, h

def construct_rgb(image, w, h):
  r = np.zeros((w, h))
  g = np.zeros((w, h))
  b = np.zeros((w, h))
  for i in range(w):
    for j in range(h):
      r[i][j] = image[i][j][0]
      g[i][j] = image[i][j][1]
      b[i][j] = image[i][j][2]

  return r, g, b

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def gen_p_q():
  p = gen_prime()
  q = gen_prime()
  while(p==q):
    q = gen_prime()

  return p,q

def calculate_totient_n(p,q):
  totient = (p-1)*(q-1)
  n = p*q
  return totient, n

def gen_public_key(totient):
  e = random.randrange(1, totient)
  while (gcd(e, totient) != 1):
    e = random.randrange(1, totient)

  return e

def gen_private_key(e, totient):
    d = None
    k = 1
    exit = False
    while not exit:
        temp = totient * k + 1
        d = float(temp/e)
        d_int = int(d)
        k += 1
        if(d_int == d):
            exit=True
    return int(d)

def encryption(m, e, n):
    m_cipher = np.copy(m)
    m_key = np.empty_like(m)
    for i in range(len(m)):
      for j in range(len(m[0])):
        m_cipher[i][j] = pow(int(m[i][j]), e, n)
        m_key[i][j] = m_cipher[i][j] // 256
        m_cipher[i][j] %= 256

    return m_cipher, m_key

def combine_channels(r, g, b, w, h):
    combined_image = np.zeros((w, h, 3))
    combined_image[:, :, 0] = r
    combined_image[:, :, 1] = g
    combined_image[:, :, 2] = b
    return combined_image

def gen_keypair():
  p, q = gen_p_q()
  totient, n = calculate_totient_n(p, q)
  e = gen_public_key(totient)
  d = gen_private_key(e, totient)

  return e, d, n

def RSA_encrypt(file_path,e,n):
  plain_img = read_image(file_path)
  w, h = image_shape(plain_img)
  r, g, b = construct_rgb(plain_img,w,h)

  r, key_r = encryption(r,e,n)
  np.save('key_r.npy', key_r)

  g, key_g = encryption(g,e,n)
  np.save('key_g.npy', key_g)

  b, key_b = encryption(b,e,n)
  np.save('key_b.npy', key_b)


  combined_image = combine_channels(r, g, b, w, h) / 255.0

  key_r_path = 'key_r.npy'
  key_g_path = 'key_g.npy'
  key_b_path = 'key_b.npy'

  return combined_image, key_r_path, key_g_path, key_b_path


def decryption(m, k, d, n):
    m_plain = np.copy(m)
    m_key = np.copy(k)
    for i in range(len(m)):
      for j in range(len(m[0])):
        m_plain[i][j] = m_key[i][j] * 256 + m_plain[i][j]
        m_plain[i][j] = pow(int(m_plain[i][j]),d,n)

    return m_plain

def RSA_decrypt(m, kr, kg, kb, d, n):

    cipher_img = m * 255.0
    w, h = image_shape(cipher_img)
    r, g, b = construct_rgb(cipher_img,w,h)

    key_r = np.load(kr)
    key_b = np.load(kb)
    key_g = np.load(kg)

    r_plain = decryption(r, key_r, d, n)
    g_plain = decryption(g, key_g, d, n)
    b_plain = decryption(b, key_b, d, n)

    combined_image = combine_channels(r_plain, g_plain, b_plain, w, h) / 255.0

    return combined_image



st.title('RSA Image Encryption and Decryption Simulation')

selected = option_menu(
   menu_title = None,
   options = ["Generate Keypair", "Encryption & Decryption"],
   icons = ["key","lock"],
   menu_icon = "cast",
   default_index = 0,
   orientation = "horizontal",
)

if selected == "Generate Keypair":
   if st.button("Generate Key Pair"):
        e, d, n = gen_keypair()
        st.success(f"Public Key (e): {e}")
        st.success(f"Private Key (d): {d}")
        st.success(f"n: {n}")
if selected == "Encryption & Decryption":
   # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # Public key input
    e = st.number_input("Enter public key (e)", min_value=0, step=1, value=0)
    n = st.number_input("Enter n", min_value=0, step=1, value=0)
    d = st.number_input("Enter private key (d)", min_value=0, step=1, value=0)

    if 'button_clicked' not in st.session_state:
      st.session_state.button_clicked = False

    encryption_completed = False
    if uploaded_file and st.button("Process"):
        st.session_state.button_clicked = True

        encrypted_image, key_r_path, key_g_path, key_b_path = RSA_encrypt(uploaded_file, e, n)

        st.image(encrypted_image, caption="Encrypted Image", use_column_width=True)

        encrypted_image_path = 'encrypted_image.jpg'
        plt.imsave(encrypted_image_path, encrypted_image, format='jpg')

        decrypted_image = RSA_decrypt(encrypted_image, key_r_path, key_g_path, key_b_path, d, n)
        st.image(decrypted_image, caption="Decrypted Image", use_column_width=True)

        decrypted_image_path = 'decrypted_image.jpg'
        plt.imsave(decrypted_image_path, decrypted_image, format='jpg')

        with zipfile.ZipFile('encrypted_data.zip', 'w') as zip_file:
          zip_file.write('encrypted_image.jpg')
          zip_file.write('decrypted_image.jpg')
          zip_file.write(key_r_path)
          zip_file.write(key_g_path)
          zip_file.write(key_b_path)

        encryption_completed = True

    if encryption_completed:
      with open('encrypted_data.zip', 'rb') as zip_file:
          st.download_button(
              label="Download Data",
              data=zip_file,
              file_name='encrypted_data.zip',
              mime='application/zip',
          )



        
    
