import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-auth.js";
import { getDatabase } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-database.js";
import { getStorage } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-storage.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-firestore.js";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDcwfi9b9WRLeTl-z038poGDfJv-J6iKlk",
  authDomain: "apphtml-ce846.firebaseapp.com",
  databaseURL: "https://apphtml-ce846-default-rtdb.firebaseio.com",
  projectId: "apphtml-ce846",
  storageBucket: "apphtml-ce846.appspot.com",
  messagingSenderId: "362316338686",
  appId: "1:362316338686:web:b1bf51c01f2ad90827dcaf",
  measurementId: "G-XXJMYK0P70"
};
// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const database = getDatabase(app);
const storage = getStorage(app);
const firestore = getFirestore(app);

export { app, auth, database, storage, firestore };
