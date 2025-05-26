import cv2
import numpy as np
from .hand_detector import HandDetector

class LSFRecognizer:
    def __init__(self):
        self.hand_detector = HandDetector()
        # Dictionnaire des signes connus
        self.known_signs = {
            "bonjour": self._is_bonjour,
            "merci": self._is_merci,
            "oui": self._is_oui,
            "non": self._is_non,
            "au_revoir": self._is_au_revoir,
            "poing_ferme": self._is_poing_ferme,
            "s_il_vous_plait": self._is_s_il_vous_plait,
            "je_t_aime": self._is_je_t_aime,
            "bien": self._is_bien,
            "manger": self._is_manger,
            "aide": self._is_aide,
            "attendre": self._is_attendre,
            "comprendre": self._is_comprendre,
            "faim": self._is_faim,
            "fatigue": self._is_fatigue,
            "dormir": self._is_dormir,
            "boire": self._is_boire,
            "froid": self._is_froid,
            "chaud": self._is_chaud,
            "pardon": self._is_pardon,
            "aujourd_hui": self._is_aujourd_hui,
            "demain": self._is_demain
        }

    def recognize_sign(self, frame):
        """
        Reconnaît le signe LSF dans l'image
        """
        # Détecter la main
        frame_with_landmarks, landmarks = self.hand_detector.detect_hand(frame)
        
        if landmarks is None:
            return "Pas de main détectée"

        # Vérifier chaque signe connu
        for sign_name, check_function in self.known_signs.items():
            if check_function(landmarks):
                return sign_name

        return "Signe non reconnu"

    def _is_bonjour(self, landmarks):
        """
        Vérifie si le signe est "Bonjour"
        """
        # Logique simplifiée : vérifie si l'index et le majeur sont levés
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return index_tip.y < landmarks.landmark[6].y and middle_tip.y < landmarks.landmark[10].y

    def _is_merci(self, landmarks):
        """
        Vérifie si le signe est "Merci"
        """
        # Logique simplifiée : vérifie si le pouce est levé
        thumb_tip = landmarks.landmark[4]
        return thumb_tip.y < landmarks.landmark[3].y

    def _is_oui(self, landmarks):
        """
        Vérifie si le signe est "Oui"
        """
        # Logique simplifiée : vérifie si l'index est levé
        index_tip = landmarks.landmark[8]
        return index_tip.y < landmarks.landmark[6].y

    def _is_non(self, landmarks):
        """
        Vérifie si le signe est "Non"
        """
        # Logique simplifiée : vérifie si l'index fait un mouvement horizontal
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        return abs(index_tip.x - index_mcp.x) > 0.1

    def _is_au_revoir(self, landmarks):
        """
        Vérifie si le signe est "Au revoir"
        """
        # Logique simplifiée : vérifie si la main est ouverte
        pinky_tip = landmarks.landmark[20]
        pinky_mcp = landmarks.landmark[17]
        return pinky_tip.y < pinky_mcp.y

    def _is_poing_ferme(self, landmarks):
        """
        Vérifie si le signe est "Poing fermé"
        """
        # Logique simplifiée : vérifie si tous les doigts sont repliés
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        finger_mcps = [5, 9, 13, 17]   # Points de référence
        return all(landmarks.landmark[tip].y > landmarks.landmark[mcp].y 
                  for tip, mcp in zip(finger_tips, finger_mcps))

    def _is_s_il_vous_plait(self, landmarks):
        """
        Vérifie si le signe est "S'il vous plaît"
        """
        # Vérifie si la main est plate et tournée vers le haut
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y < wrist.y and abs(middle_tip.x - wrist.x) < 0.1

    def _is_je_t_aime(self, landmarks):
        """
        Vérifie si le signe est "Je t'aime"
        """
        # Vérifie si l'index et le majeur sont croisés
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return abs(index_tip.x - middle_tip.x) < 0.05 and abs(index_tip.y - middle_tip.y) < 0.05

    def _is_bien(self, landmarks):
        """
        Vérifie si le signe est "Bien"
        """
        # Vérifie si le pouce est levé et les autres doigts sont repliés
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        finger_mcps = [5, 9, 13, 17]   # Points de référence
        return (thumb_tip.y < thumb_mcp.y and 
                all(landmarks.landmark[tip].y > landmarks.landmark[mcp].y 
                    for tip, mcp in zip(finger_tips, finger_mcps)))

    def _is_manger(self, landmarks):
        """
        Vérifie si le signe est "Manger"
        """
        # Vérifie si la main est près du visage
        wrist = landmarks.landmark[0]
        return wrist.y < 0.3  # La main doit être dans la partie supérieure de l'image

    def _is_aide(self, landmarks):
        """
        Vérifie si le signe est "Aide"
        """
        # Vérifie si la main est ouverte et tournée vers le haut
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        return all(landmarks.landmark[tip].y < wrist.y for tip in finger_tips)

    def _is_attendre(self, landmarks):
        """
        Vérifie si le signe est "Attendre"
        """
        # Vérifie si la main est immobile (position neutre)
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) < 0.05 and abs(middle_tip.y - wrist.y) < 0.05

    def _is_comprendre(self, landmarks):
        """
        Vérifie si le signe est "Comprendre"
        """
        # Vérifie si l'index est près du front
        index_tip = landmarks.landmark[8]
        return index_tip.y < 0.2  # La main doit être dans la partie supérieure de l'image

    def _is_faim(self, landmarks):
        """
        Vérifie si le signe est "Faim"
        """
        # Vérifie si la main est dans la partie centrale de l'image
        wrist = landmarks.landmark[0]
        return 0.3 < wrist.y < 0.7  # La main doit être dans la partie centrale

    def _is_fatigue(self, landmarks):
        """
        Vérifie si le signe est "Fatigué"
        """
        # Vérifie si la main est près du front
        wrist = landmarks.landmark[0]
        return wrist.y < 0.25  # La main doit être dans la partie supérieure de l'image

    def _is_dormir(self, landmarks):
        """
        Vérifie si le signe est "Dormir"
        """
        # Vérifie si la main est près de la joue
        wrist = landmarks.landmark[0]
        return wrist.y < 0.3 and abs(wrist.x - 0.5) < 0.2  # Main près du visage

    def _is_boire(self, landmarks):
        """
        Vérifie si le signe est "Boire"
        """
        # Vérifie si la main est près de la bouche
        wrist = landmarks.landmark[0]
        return wrist.y < 0.35 and abs(wrist.x - 0.5) < 0.15  # Main près de la bouche

    def _is_froid(self, landmarks):
        """
        Vérifie si le signe est "Froid"
        """
        # Vérifie si les doigts sont légèrement écartés
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        return all(abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.05 
                  for tip in finger_tips)

    def _is_chaud(self, landmarks):
        """
        Vérifie si le signe est "Chaud"
        """
        # Vérifie si la main est ouverte et les doigts sont écartés
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return all(landmarks.landmark[tip].y < wrist.y and 
                  abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.1 
                  for tip in finger_tips)

    def _is_pardon(self, landmarks):
        """
        Vérifie si le signe est "Pardon"
        """
        # Vérifie si la main fait un mouvement circulaire
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) > 0.1 and abs(middle_tip.y - wrist.y) > 0.1

    def _is_aujourd_hui(self, landmarks):
        """
        Vérifie si le signe est "Aujourd'hui"
        """
        # Vérifie si l'index pointe vers le bas
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        return index_tip.y > index_mcp.y and abs(index_tip.x - index_mcp.x) < 0.05

    def _is_demain(self, landmarks):
        """
        Vérifie si le signe est "Demain"
        """
        # Vérifie si l'index pointe vers l'avant
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        return abs(index_tip.y - index_mcp.y) < 0.05 and index_tip.x > index_mcp.x

# Instance globale du reconnaisseur
recognizer = LSFRecognizer()

def recognize_lsf_sign(frame):
    """
    Fonction utilitaire pour reconnaître un signe LSF
    """
    return recognizer.recognize_sign(frame) 