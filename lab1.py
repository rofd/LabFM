#Radiation Intensity vs. Wavelength for different temperatures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

# Constants
h = 6.626e-34  # Planck's constant (Joule seconds)
c = 3.0e8      # Speed of light (meters per second)
k = 1.38e-23   # Boltzmann constant (Joules per Kelvin)

# Wavelength range in micrometers
wavelengths_um = np.linspace(0.1, 3, 100)  # Wavelengths in micrometers

# Temperatures
temperatures_K = [2000, 3000, 4000, 5000]  # Temperatures in Kelvin

# Function to calculate radiation intensity
def radiation_intensity(wavelength, temperature):
    return (2 * h * c**2) / (wavelength**5) * (1 / (np.exp((h * c) / (wavelength * k * temperature)) - 1))

# Find the wavelength corresponding to the maximum intensity for each temperature
max_wavelengths = []
max_intensities = []
for temp in temperatures_K:
    intensity = [radiation_intensity(w * 1e-6, temp) for w in wavelengths_um]
    max_wavelength_index = np.argmax(intensity)
    max_wavelength = wavelengths_um[max_wavelength_index]
    max_intensity = intensity[max_wavelength_index]
    max_wavelengths.append(max_wavelength)
    max_intensities.append(max_intensity)

# Interpolate to get a curve that joins the maximum intensity points
interp_function = interp1d(max_wavelengths, max_intensities, kind='quadratic')
interp_wavelengths = np.linspace(min(max_wavelengths), max(max_wavelengths), 100)
interp_intensities = interp_function(interp_wavelengths)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the curves for different temperatures
for i, temp in enumerate(temperatures_K):
    intensity = [radiation_intensity(w * 1e-6, temp) for w in wavelengths_um]
    ax.plot(wavelengths_um, intensity, label=f"{temp} K")

    # Add markers for the maximum points
    ax.plot(max_wavelengths[i], max_intensities[i], 'o', color='black')

# Plot the interpolated curve
ax.plot(interp_wavelengths, interp_intensities, 'k--', label='Curva interpolada')

# Tufte's recommendations for minimalistic design
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1e12))
ax.legend()

# Labels and title
ax.set_xlabel("Longitud de Onda (μm)")
ax.set_ylabel("Intensidad de Radiación (W/m²/μm)")
#ax.set_title("Intensidad de Radiación vs. Longitud de nda para diferenetes temperaturas")

# Define visible wavelength range and corresponding colors
visible_range = [0.38, 0.45, 0.50, 0.56, 0.60, 0.65, 0.75]  # Visible wavelength range in micrometers

# Define colors for the visible spectrum
colors = ['violet', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']

# Shade the visible spectrum regions
for i in range(len(colors)):
    if i < len(colors) - 1:
        ax.axvspan(visible_range[i], visible_range[i+1], color=colors[i], alpha=0.05)

plt.tight_layout()
plt.show()


#Millivolt readings for different cube face materials
import matplotlib.pyplot as plt

# Provided data for mV readings, surface types, and updated temperature values
surface_types = ['Cara Negra', 'Cara Blanca', 'Aluminio Pulido', 'Aluminio Opaco']
mV_readings = {
    'Cara Negra': [7.6, 6.6, 6.1, 4.9, 4.3],
    'Cara Blanca': [7.6, 6.0, 5.7, 0.5, 4.6],
    'Aluminio Pulido': [0.4, 0.3, 0.2, 0.2, 0.2],
    'Aluminio Opaco': [1.8, 1.4, 1.3, 1.1, 1.1]
}
# Updated temperature values for each mV reading
temperature_values = [80, 81, 76, 67, 65]

# Create a bar plot for mV readings with temperature
plt.figure(figsize=(10, 6))

# Defining a list of colors for each surface type
colors = ['blue', 'orange', 'green', 'red']

# Creating the bar plot for all surface types
for i, surface in enumerate(surface_types):
    x_values = [t + (i - 1.5) * 0.15 for t in temperature_values]  # Adjust positioning of bars
    plt.bar(x_values, mV_readings[surface], color=colors[i], width=0.15, align='center', alpha=0.6, label=surface)
    # Adding millivolt measurements on top of each bar
    for j, mV in enumerate(mV_readings[surface]):
        plt.text(x_values[j], mV + 0.2, str(mV), ha='center', fontsize=9, color='black')

# Adding labels and title
plt.xlabel('Temperatura (°C)')
plt.ylabel('Milivolts (mV)')
#plt.title('Lecturas de milivolts para diferentes materiales de las caras del cubo')

# Remove spines and ticks for Tufte style
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

# Adding a legend
plt.legend()

# Display the plot
plt.grid(False)
plt.tight_layout()
plt.xticks(rotation=0, ha='center')  # Rotate x-axis labels for better visibility
plt.show()


#Scatter plot and fitted theoretical results of the inverse square law of light
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example dataset: Replace with your actual data
variable1 = [2, 3, 4, 6, 7, 8, 9, 10, 15, 17, 20, 22.5, 25, 27.5, 30, 32.5, 35, 40, 45]
variable2 = [101.6, 43.4, 32.1, 15.6, 13.5, 11.3, 7.4, 7.1, 4.6, 3.0, 2.4, 1.9, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.3]

# Inverse square law function
def inverse_square_law(r, I0):
    return I0 / (r ** 2)

# Fit the inverse square law function to the data
params, covariance = curve_fit(inverse_square_law, variable1, variable2)

# Extract the optimized parameter I0
I0_fit = params[0]

# Calculate theoretical intensities based on the fitted I0
theoretical_intensities_fit = [inverse_square_law(r, I0_fit) for r in variable1]

# Create a scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(variable1, variable2, c='blue', alpha=0.7, label='Mediciones Experimentales')
plt.plot(variable1, theoretical_intensities_fit, c='green', linestyle='--', label='Resultado Teórico (Ajustado)')

# Add grid lines
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Remove spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add labels and title
plt.xlabel('Distancia (r)')
plt.ylabel('Intensidad (I)')
#plt.title('Gráfico de dispersión y resultados teóricos ajustados de la ley del cuadrado inverso de la luz')

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


#Measured Potential Difference vs. Temperature Difference Fourth Power with Linear Fit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Given data
temperature = np.array([80, 81, 76, 67, 65])  # Temperature in °C
resistance_black = np.array([11.22, 10.45, 12.37, 15.96, 18.18])  # Resistance for black face in kΩ

# Function to calculate the temperature difference's fourth power
def temp_diff_power(t, t_ref):
    return (t**4) - (t_ref**4)

# Calculate the temperature difference's fourth power using a reference temperature T0
T0 = 20  # Reference temperature in °C
temp_diff_power_values = temp_diff_power(temperature, T0)

# Measured potential difference at the output of the radiation sensor (hypothetical values)
measured_potential_difference = np.array([0.12, 0.08, 0.05, 0.03, 0.02])  # Hypothetical values

# Linear fit using numpy.polyfit()
slope, intercept = np.polyfit(temp_diff_power_values, measured_potential_difference, 1)

# Calculate the fitted line
fitted_line = slope * temp_diff_power_values + intercept

# Tufte-style plot
plt.figure(figsize=(8, 6))

# Scatter plot
plt.scatter(temp_diff_power_values, measured_potential_difference, marker='o', color='blue', label='Mediciones experimentales')

# Plot the linear fit
plt.plot(temp_diff_power_values, fitted_line, color='red', linestyle='--', label='Ajuste lineal')

# Adding labels and title
plt.xlabel('Diferencia de temperatura Cuarta potencia ($T_1^4 - T_0^4$)')
plt.ylabel('Diferencia de potencial')
#plt.title("Measured Potential Difference vs. Temperature Difference Fourth Power with Linear Fit")

# Adding a legend
plt.legend()

# Removing the top and right spines of the axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show the plot
plt.tight_layout()
plt.show()
