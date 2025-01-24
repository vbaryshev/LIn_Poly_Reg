/*
  Пример приложения на C++ с визуализацией регрессии (линейной и полиномиальной 2-й степени),
  пользовательским вводом значения X, добавлением новых точек, удалением ближней точки
  и прочими улучшениями:

    - Динамическая подстройка под изменение размеров окна SFML.
    - Сохранение новых точек в data_updated.csv (по нажатию S).
    - Подсветка "дальних" точек (по разнице с текущей моделью регрессии).
    - Отображение координат (X,Y) в области данных около курсора.
    - Выбор между линейной регрессией и полиномиальной (2-й степени) нажатием клавиш 1 и 2.

  Используется библиотека SFML для графики.

  Сборка (Ubuntu, например):
      g++ main.cpp -o ImprovedLinRegGUI -lsfml-graphics -lsfml-window -lsfml-system

  Запуск:
      ./ImprovedLinRegGUI

  Требуется наличие файлов:
      1) data.csv    - CSV-файл с начальными точками (X, Y).
      2) arial.ttf   - файл шрифта (для отрисовки текста).
*/

#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <cstring> // <-- Добавьте этот заголовок для memcpy


// Структура, чтобы хранить обучающие точки (X, Y)
struct Point
{
    float x;
    float y;
};

// Тип регрессии: линейная или полиномиальная (2-й степени)
enum class RegressionType
{
    LINEAR,
    POLYNOMIAL2
};

// Функция считывания CSV
std::vector<Point> loadDataFromCSV(const std::string& filename)
{
    std::vector<Point> data;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) 
            continue;

        std::stringstream ss(line);
        float xVal, yVal;
        char delimiter;
        // Попробуем считать x и y
        if (ss >> xVal)
        {
            if (ss.peek() == ',' || ss.peek() == ';')
                ss >> delimiter;
            if (ss >> yVal)
            {
                data.push_back({xVal, yVal});
            }
        }
    }
    file.close();
    return data;
}

// Функция сохранения данных в CSV
void saveDataToCSV(const std::string& filename, const std::vector<Point>& dataPoints)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open save file " << filename << std::endl;
        return;
    }

    for (auto& p : dataPoints)
        file << p.x << "," << p.y << "\n";

    file.close();
    std::cout << "Data saved to " << filename << std::endl;
}

// ----------------------------------------
// Линейная регрессия (y = slope*x + intercept)
// ----------------------------------------
std::pair<float, float> computeLinearRegression(const std::vector<Point>& points)
{
    if (points.empty())
    {
        return {0.f, 0.f};
    }

    float sumX = 0.f, sumY = 0.f;
    for (auto& p : points)
    {
        sumX += p.x;
        sumY += p.y;
    }
    float meanX = sumX / points.size();
    float meanY = sumY / points.size();

    float numerator = 0.f;
    float denominator = 0.f;
    for (auto& p : points)
    {
        float dx = p.x - meanX;
        float dy = p.y - meanY;
        numerator   += dx * dy;
        denominator += dx * dx;
    }

    float slope = 0.f;
    if (denominator != 0.f)
        slope = numerator / denominator;
    float intercept = meanY - slope * meanX;

    return {slope, intercept};
}

// ----------------------------------------
// Полиномиальная регрессия 2-й степени: y = a*x^2 + b*x + c
// ----------------------------------------
struct Poly2Coeffs
{
    float a; // при x^2
    float b; // при x
    float c; // свободный член
};

// Решим систему для a,b,c методом составления нормальных уравнений и их решения
// Формулой Крамера или через матрицу 3x3 (классическая M^-1 * R).
Poly2Coeffs computePolynomialRegression2(const std::vector<Point>& points)
{
    Poly2Coeffs coeffs{0.f, 0.f, 0.f};
    if (points.size() < 3)
    {
        // Для корректной аппроксимации 2-й степенью нужно хотя бы 3 точки
        return coeffs;
    }

    // Суммы
    double Sx   = 0.0;
    double Sy   = 0.0;
    double Sx2  = 0.0;
    double Sx3  = 0.0;
    double Sx4  = 0.0;
    double Sxy  = 0.0;
    double Sx2y = 0.0;
    int n = static_cast<int>(points.size());

    for (auto& p : points)
    {
        double x  = p.x;
        double y  = p.y;
        double x2 = x*x;
        double x3 = x2*x;
        double x4 = x3*x;

        Sx   += x;
        Sy   += y;
        Sx2  += x2;
        Sx3  += x3;
        Sx4  += x4;
        Sxy  += x*y;
        Sx2y += x2*y;
    }

    // Матрица (3x3) и вектор правой части:
    // [ n    Sx   Sx2  ] [ c ] = [ Sy   ]
    // [ Sx   Sx2  Sx3  ] [ b ]   [ Sxy  ]
    // [ Sx2  Sx3  Sx4  ] [ a ]   [ Sx2y ]
    //
    // Ищем (a, b, c).

    // Составим матрицу A и вектор B
    double A[3][3] = {
        { (double)n,  Sx,    Sx2 },
        { Sx,         Sx2,   Sx3 },
        { Sx2,        Sx3,   Sx4 }
    };
    double B[3] = { Sy, Sxy, Sx2y };

    // Чтобы не тащить целую библиотеку линалгебры, можно решить через Крамера вручную:
    auto det3 = [&](double m[3][3]){
        return m[0][0]* (m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
               m[0][1]* (m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
               m[0][2]* (m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    };

    double D = det3(A);
    if (std::fabs(D) < 1e-12)
    {
        // Матрица вырождена; иногда такое бывает при плохих данных.
        // Вернём нули.
        return coeffs;
    }

    // D_c (для c)
    double A_c[3][3];
    std::memcpy(A_c, A, 9*sizeof(double));
    A_c[0][0] = B[0];
    A_c[1][0] = B[1];
    A_c[2][0] = B[2];
    double D_c = det3(A_c);

    // D_b (для b)
    double A_b[3][3];
    std::memcpy(A_b, A, 9*sizeof(double));
    A_b[0][1] = B[0];
    A_b[1][1] = B[1];
    A_b[2][1] = B[2];
    double D_b = det3(A_b);

    // D_a (для a)
    double A_a[3][3];
    std::memcpy(A_a, A, 9*sizeof(double));
    A_a[0][2] = B[0];
    A_a[1][2] = B[1];
    A_a[2][2] = B[2];
    double D_a = det3(A_a);

    double c = D_c / D;
    double b = D_b / D;
    double a = D_a / D;
    coeffs.a = static_cast<float>(a);
    coeffs.b = static_cast<float>(b);
    coeffs.c = static_cast<float>(c);

    return coeffs;
}

// Вспомогательная функция для вычисления значения полинома 2-й степени
float evaluatePoly2(const Poly2Coeffs& coeffs, float x)
{
    return coeffs.a*x*x + coeffs.b*x + coeffs.c;
}

int main()
{
    // -----------------------------
    // 1. Загрузка / подготовка данных
    // -----------------------------
    std::string csvFile = "data.csv";
    std::vector<Point> dataPoints = loadDataFromCSV(csvFile);
    if (dataPoints.empty())
    {
        std::cerr << "Empty or invalid data. Using demo data..." << std::endl;
        // Если данных нет - подставим демо
        dataPoints.push_back({1.f, 1.f});
        dataPoints.push_back({2.f, 2.f});
        dataPoints.push_back({3.f, 1.3f});
        dataPoints.push_back({4.f, 3.f});
        dataPoints.push_back({5.f, 4.5f});
    }

    // Окно
    sf::RenderWindow window(sf::VideoMode(800, 600), "Regression Linear or Polinom");
    window.setFramerateLimit(60);

    // Шрифт
    sf::Font font;
    if (!font.loadFromFile("arial.ttf"))
    {
        std::cerr << "Error: Could not load font arial.ttf" << std::endl;
        return 1;
    }

    // Текст ввода X
    std::string userInputX;
    sf::Text inputPrompt("Enter X value (Press Enter):", font, 16);
    inputPrompt.setFillColor(sf::Color::White);
    inputPrompt.setPosition(20.f, 20.f);

    sf::Text inputText("", font, 16);
    inputText.setFillColor(sf::Color::Yellow);
    inputText.setPosition(20.f, 50.f);

    // Текст предсказанного Y
    sf::Text predictionText("Prediction: Y = ?", font, 16);
    predictionText.setFillColor(sf::Color::Red);
    predictionText.setPosition(20.f, 80.f);

    // Подсказка (мышь, сохранение, выбор режима регрессии)
    sf::Text mouseHint("LMB=add point; RMB=remove; S=save; l=Linear; p=Poly2", font, 16);
    mouseHint.setFillColor(sf::Color::White);
    mouseHint.setPosition(400.f, 20.f);

    // Текст с текущим типом регрессии
    sf::Text regTypeText("Regression Linear", font, 16);
    regTypeText.setFillColor(sf::Color::Magenta);
    regTypeText.setPosition(400.f, 50.f);

    // Текст для отображения координат около курсора
    sf::Text mouseCoordsText("", font, 14);
    mouseCoordsText.setFillColor(sf::Color::White);

    // Параметры линейной регрессии
    float slope = 0.f;
    float intercept = 0.f;

    // Параметры полиномиальной регрессии 2-й степени
    Poly2Coeffs polyCoeffs{0.f, 0.f, 0.f};

    // Какой тип регрессии используем сейчас
    RegressionType currentReg = RegressionType::LINEAR;

    // Границы по X и Y
    float minX = 0.f, maxX = 0.f;
    float minY = 0.f, maxY = 0.f;

    // Оси
    sf::VertexArray axisX(sf::Lines, 2);
    sf::VertexArray axisY(sf::Lines, 2);
    sf::Text labelX("X", font, 16);
    sf::Text labelY("Y", font, 16);

    // ------------------------------------
    // Лямбда для обновления модели и границ
    // ------------------------------------
    auto updateModelAndBounds = [&]()
    {
        if (!dataPoints.empty())
        {
            // Находим min/max
            minX = dataPoints.front().x;
            maxX = dataPoints.front().x;
            minY = dataPoints.front().y;
            maxY = dataPoints.front().y;
            for (auto& p : dataPoints)
            {
                if (p.x < minX) minX = p.x;
                if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }
            // Добавим небольшой отступ
            float pad = 1.f;
            minX -= pad; maxX += pad;
            minY -= pad; maxY += pad;

            // Пересчитываем модель по выбранному типу регрессии
            if (currentReg == RegressionType::LINEAR)
            {
                auto [s, b] = computeLinearRegression(dataPoints);
                slope = s;
                intercept = b;
            }
            else // POLYNOMIAL2
            {
                polyCoeffs = computePolynomialRegression2(dataPoints);
            }
        }
        else
        {
            // Если список пуст, ставим что-то по умолчанию
            minX = -1.f; maxX = 1.f;
            minY = -1.f; maxY = 1.f;
            // Линейные коэффициенты
            slope = 0.f; intercept = 0.f;
            // Полиномиальные
            polyCoeffs = {0.f, 0.f, 0.f};
        }
    };

    // Изначальный пересчёт
    updateModelAndBounds();

    // Преобразования координат
    auto toScreenCoords = [&](float x, float y)
    {
        float w = static_cast<float>(window.getSize().x);
        float h = static_cast<float>(window.getSize().y);

        float topMargin = 120.f;

        float screenX = 50.f + (x - minX) / (maxX - minX) * (w - 100.f);
        float screenY = h - 50.f - (y - minY) / (maxY - minY) * (h - topMargin - 100.f);

        return sf::Vector2f(screenX, screenY);
    };

    auto toDataCoords = [&](float sx, float sy)
    {
        float w = static_cast<float>(window.getSize().x);
        float h = static_cast<float>(window.getSize().y);
        float topMargin = 120.f;

        float x = minX + (sx - 50.f) / (w - 100.f) * (maxX - minX);

        float normY = ((h - 50.f) - sy) / (h - topMargin - 100.f);
        float y = minY + normY * (maxY - minY);

        return sf::Vector2f(x, y);
    };
    // Обновление осей
    auto updateAxes = [&]()
    {
        // Ось X (из (minX, 0) в (maxX, 0))
        axisX[0].position = toScreenCoords(minX, 0.f);
        axisX[1].position = toScreenCoords(maxX, 0.f);
        axisX[0].color = sf::Color::White;
        axisX[1].color = sf::Color::White;

        // Ось Y (из (0, minY) в (0, maxY))
        axisY[0].position = toScreenCoords(0.f, minY);
        axisY[1].position = toScreenCoords(0.f, maxY);
        axisY[0].color = sf::Color::White;
        axisY[1].color = sf::Color::White;

        // Подпись X
        sf::Vector2f lx = toScreenCoords(maxX, 0.f);
        labelX.setString("X");
        labelX.setFillColor(sf::Color::White);
        labelX.setPosition(lx.x - 20.f, lx.y + 5.f);

        // Подпись Y
        sf::Vector2f ly = toScreenCoords(0.f, maxY);
        labelY.setString("Y");
        labelY.setFillColor(sf::Color::White);
        labelY.setPosition(ly.x + 5.f, ly.y);
    };

    // Первый вызов
    updateAxes();

    // Функция удаления ближайшей точки
    auto removeNearestPoint = [&](float mouseXScreen, float mouseYScreen) {
        if (dataPoints.empty()) return;

        float minDist = std::numeric_limits<float>::max();
        int minIndex = -1;

        for (int i = 0; i < static_cast<int>(dataPoints.size()); ++i)
        {
            sf::Vector2f ptScreen = toScreenCoords(dataPoints[i].x, dataPoints[i].y);
            float dx = ptScreen.x - mouseXScreen;
            float dy = ptScreen.y - mouseYScreen;
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist < minDist)
            {
                minDist = dist;
                minIndex = i;
            }
        }

        // Порог 10 px
        if (minDist < 10.f && minIndex >= 0)
        {
            dataPoints.erase(dataPoints.begin() + minIndex);
            updateModelAndBounds();
            updateAxes();
        }
    };

    // Основной цикл
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Закрытие
            if (event.type == sf::Event::Closed)
                window.close();

            // Ресайз
            if (event.type == sf::Event::Resized)
            {
                sf::FloatRect visibleArea(0.f, 0.f, event.size.width, event.size.height);
                window.setView(sf::View(visibleArea));
                updateAxes();
            }

            // Ввод текста (для поля userInputX)
            if (event.type == sf::Event::TextEntered)
            {
                // Enter
                if (event.text.unicode == '\r' || event.text.unicode == '\n')
                {
                    // Преобразуем введённый X в число и считаем предсказание
                    try {
                        float xVal = std::stof(userInputX);
                        float yPred = 0.f;
                        if (currentReg == RegressionType::LINEAR)
                        {
                            yPred = slope * xVal + intercept;
                        }
                        else
                        {
                            yPred = evaluatePoly2(polyCoeffs, xVal);
                        }
                        predictionText.setString("Prediction: Y = " + std::to_string(yPred));
                    }
                    catch (...)
                    {
                        predictionText.setString("Prediction: invalid X");
                    }
                }
                // Backspace
                else if (event.text.unicode == 8 && !userInputX.empty())
                {
                    userInputX.pop_back();
                }
                // Печатные символы
                else if (event.text.unicode < 128)
                {
                    char enteredChar = static_cast<char>(event.text.unicode);
                    if ((enteredChar >= '0' && enteredChar <= '9') ||
                        enteredChar == '.' || enteredChar == '-' )
                    {
                        userInputX += enteredChar;
                    }
                }
            }

            // Нажатия клавиш
            if (event.type == sf::Event::KeyPressed)
            {
                // Сохранение CSV
                if (event.key.code == sf::Keyboard::S)
                {
                    saveDataToCSV("data_updated.csv", dataPoints);
                }
                // Выбор линейной регрессии
                if (event.key.code == sf::Keyboard::L)
                {
                    currentReg = RegressionType::LINEAR;
                    regTypeText.setString("Current Regression: Linear");
                    updateModelAndBounds();
                    updateAxes();
                }
                // Выбор полиномиальной (2-й степени)
                if (event.key.code == sf::Keyboard::P)
                {
                    currentReg = RegressionType::POLYNOMIAL2;
                    regTypeText.setString("Regression Polynomial (2nd degree)");
                    updateModelAndBounds();
                    updateAxes();
                }
            }

            // Мышь
            if (event.type == sf::Event::MouseButtonPressed)
            {
                float sx = static_cast<float>(event.mouseButton.x);
                float sy = static_cast<float>(event.mouseButton.y);

                if (event.mouseButton.button == sf::Mouse::Left)
                {
                    // Добавить точку
                    sf::Vector2f dataPos = toDataCoords(sx, sy);
                    dataPoints.push_back({dataPos.x, dataPos.y});
                    updateModelAndBounds();
                    updateAxes();
                }
                else if (event.mouseButton.button == sf::Mouse::Right)
                {
                    // Удалить точку
                    removeNearestPoint(sx, sy);
                }
            }
        }

        // Обновляем текст ввода
        inputText.setString(userInputX);

        // Обновляем мышиные координаты (коорд. данных), выводим рядом с курсором
        sf::Vector2i mousePos = sf::Mouse::getPosition(window);
        float mx = static_cast<float>(mousePos.x);
        float my = static_cast<float>(mousePos.y);
        sf::Vector2f dataPos = toDataCoords(mx, my);
        // Форматируем строку, например, до 2-3 знаков
        std::stringstream ss;
        ss.precision(2);
        ss << std::fixed << "X=" << dataPos.x << ", Y=" << dataPos.y;
        mouseCoordsText.setString(ss.str());
        // Разместим текст чуть правее/ниже курсора, чтобы не сливался
        mouseCoordsText.setPosition(mx + 10.f, my + 10.f);

        // Рисование
        window.clear(sf::Color(30, 30, 60));

        window.draw(inputPrompt);
        window.draw(inputText);
        window.draw(predictionText);
        window.draw(mouseHint);
        window.draw(regTypeText);

        // Оси
        window.draw(axisX);
        window.draw(axisY);
        window.draw(labelX);
        window.draw(labelY);

        // Точки
        float highlightThreshold = 0.5f;
        for (auto& p : dataPoints)
        {
            // Считаем "предсказанную" Y по текущей модели
            float yPred = 0.f;
            if (currentReg == RegressionType::LINEAR)
            {
                yPred = slope*p.x + intercept;
            }
            else
            {
                yPred = evaluatePoly2(polyCoeffs, p.x);
            }

            float diff = std::fabs(p.y - yPred);
            sf::Color ptColor = (diff > highlightThreshold) ? sf::Color::Red : sf::Color::Red;

            sf::Vector2f ptPos = toScreenCoords(p.x, p.y);
            sf::CircleShape shape(3.f);
            shape.setFillColor(ptColor);
            shape.setPosition(ptPos.x - 3.f, ptPos.y - 3.f);
            window.draw(shape);
        }

        // Рисуем регрессионную функцию
        if (!dataPoints.empty())
        {
            // Чтобы "график" полинома (или прямой) был плавным, разобьём на сегменты
            sf::VertexArray curve(sf::LineStrip);
            int segments = 200;
            for (int i = 0; i <= segments; ++i)
            {
                // t идёт от 0 до 1
                float t = static_cast<float>(i) / static_cast<float>(segments);
                float xVal = minX + t*(maxX - minX);

                float yVal = 0.f;
                if (currentReg == RegressionType::LINEAR)
                {
                    yVal = slope * xVal + intercept;
                }
                else
                {
                    yVal = evaluatePoly2(polyCoeffs, xVal);
                }

                sf::Vector2f sc = toScreenCoords(xVal, yVal);
                curve.append(sf::Vertex(sc, sf::Color::Green));
            }
            window.draw(curve);
        }

        // Рисуем текст координат у курсора
        window.draw(mouseCoordsText);

        window.display();
    }

    return 0;
}

