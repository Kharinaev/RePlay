pipeline {
    agent { node { label 'Linux_Default' } }
    stages {
        stage('Build') {
            steps {
                echo 'building package with poetry'
		sh './build_package.sh'
            }
        }
        stage('Test') {
            steps {
                echo 'testing'
		sh './test_package.sh'
            }
        }
    }
}
